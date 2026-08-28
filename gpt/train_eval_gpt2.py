import os
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import bitsandbytes as bnb
import tiktoken
from collections import OrderedDict

import gpt_dataset
from gpt import GPT
from transformer import utils


def load_state_dict_strip_module(path, map_location="cpu"):
    """Loads a state_dict and strips a leading 'module.' if present."""
    state_dict = torch.load(path, map_location=map_location)
    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    return new_state_dict


def setup():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def cleanup():
    dist.destroy_process_group()


def validate(val_dataloader, language_model, device, global_rank, vocab_len, world_size):
    language_model.eval()

    total_loss = torch.zeros(1, device=device, dtype=torch.float16)
    count = torch.zeros(1, device=device, dtype=torch.float16)
    loss_fn = nn.CrossEntropyLoss()

    if global_rank == 0:
        print('')

    step = 0
    stop_val_step = 50
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(val_dataloader):
            step += 1
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = language_model(inputs)
                loss = loss_fn(outputs.view(-1, vocab_len), labels.view(-1))

            total_loss += loss.detach().to(total_loss.dtype)
            count += 1.0

            if global_rank == 0:
                print(f"\rPerforming validation. Iteration {step}: Loss {loss.item()}", end="", flush=True)

            if step % stop_val_step == 0:
                break

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    avg = total_loss / count

    return float(avg.item())


def run_train(config, device, local_rank, global_rank, world_size):
    base_lr = config["base_lr"]
    seq_len = config["seq_len"]
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]

    iter_batch_size = config["iter_batch_size"]
    actual_gpt_batch_size = config["actual_gpt_batch_size"]
    grad_accum_steps = int(actual_gpt_batch_size/(seq_len * iter_batch_size * world_size))

    max_warmup_steps = config["max_warmup_steps"]
    val_step = config["val_step"]
    use_checkpointing = config["use_checkpointing"]
    use_flash = config["use_flash"]
    max_steps = config["max_steps"]
    min_lr = config["min_lr"]

    data_dir = config["data_dir"]
    ckpt_path = config["ckpt_path"]

    gpt_train_dataset = gpt_dataset.GPTDataset(data_dir, 'train', iter_batch_size, seq_len, local_rank, world_size)
    gpt_val_dataset = gpt_dataset.GPTDataset(data_dir, 'val', iter_batch_size, seq_len, local_rank, world_size)

    vocab_len = gpt_train_dataset.vocab_len

    train_dataloader = DataLoader(
        gpt_train_dataset,
        batch_size=None,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    val_dataloader = DataLoader(
        gpt_val_dataset,
        batch_size=None,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    language_model = GPT(
        dictionary_length=vocab_len,
        seq_len=seq_len,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_checkpointing=use_checkpointing,
        use_flash=use_flash,
    )

    language_model.apply(utils.weights_init)
    language_model = language_model.to(device)
    language_model = DDP(language_model, device_ids=[local_rank])

    decay, no_decay = utils.get_regularization_parameters(language_model, num_layers)

    optimizer = bnb.optim.AdamW8bit([
        {'params': decay, 'weight_decay': 0.1},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=base_lr)

    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()
    step = 0
    best_val_loss = float('inf')
    total_train_loss = torch.zeros(1, device=device, dtype=torch.float16)
    loss_accum_steps = torch.zeros(1, device=device, dtype=torch.float32)
    valid_batch_count = torch.zeros(1, device=device, dtype=torch.float32)

    optimizer.zero_grad()
    optimizer = utils.optimizer_warmup(optimizer, base_lr, 1, max_warmup_steps)
    first_clock = time.time()

    while True:
        if step // grad_accum_steps > max_steps:
            break
    
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            step += 1
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
    
            with torch.amp.autocast("cuda", dtype=torch.float16):
                is_sync_step = (step % grad_accum_steps == 0)
                language_model.require_backward_grad_sync = is_sync_step
    
                outputs = language_model(inputs)
                loss = loss_fn(outputs.view(-1, vocab_len), labels.view(-1))
    
                loss_accum_steps += loss.detach()
                valid_batch_count += 1.0
    
                scaled_loss = loss / grad_accum_steps
                scaler.scale(scaled_loss).backward()
    
            if is_sync_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(language_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
    
                dist.all_reduce(loss_accum_steps, op=dist.ReduceOp.SUM)
                dist.all_reduce(valid_batch_count, op=dist.ReduceOp.SUM)
    
                valid_batch_count = torch.clamp(valid_batch_count, min=1.0)
                step_avg_loss = loss_accum_steps / valid_batch_count
                total_train_loss += step_avg_loss
    
                update_step = step // grad_accum_steps
    
                if global_rank == 0:
                    elapsed_time = time.time() - first_clock
                    tokens_per_sec = (grad_accum_steps * iter_batch_size * seq_len * world_size) / elapsed_time
                    print(f"\rStep {update_step}: Loss {step_avg_loss.item():.4f}, batch time {tokens_per_sec}, grad accum steps {grad_accum_steps}", end="", flush=True)
                    first_clock = time.time()
    
                loss_accum_steps.zero_()
                valid_batch_count.zero_()
                optimizer.zero_grad()
    
                if update_step <= max_warmup_steps:
                    optimizer = utils.optimizer_warmup(optimizer, base_lr, update_step, max_warmup_steps)
                else:
                    optimizer = utils.cosine_lr_update(optimizer, update_step, base_lr, max_steps, max_warmup_steps, min_lr=min_lr)
    
                if update_step % val_step == 0:
                    val_loss = validate(
                        val_dataloader=val_dataloader,
                        language_model=language_model,
                        device=device,
                        global_rank=global_rank,
                        vocab_len=vocab_len,
                        world_size=world_size,
                    )
    
                    dist.all_reduce(total_train_loss, op=dist.ReduceOp.SUM)
                    total_train_loss /= (world_size * val_step)
    
                    if global_rank == 0:
                        print(f'\n Step {update_step}: avg train loss {total_train_loss.item()}, avg val loss {val_loss}', flush=True)
    
                    if val_loss < best_val_loss and global_rank == 0:
                        best_val_loss = val_loss
                        checkpoint = {
                            'model_state_dict': language_model.module.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'step': step,
                            'best_val_loss': best_val_loss,
                        }
                        # torch.save(checkpoint, ckpt_path)
    
                    total_train_loss.zero_()
                    dist.barrier(device_ids=[local_rank])
                    language_model.train()
    
                if update_step > max_steps:
                    break


def run_test(config, device, local_rank, global_rank, world_size):
    seq_len = config["seq_len"]
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    use_checkpointing = config["use_checkpointing"]
    use_flash = config["use_flash"]
    ckpt_path = config["ckpt_path"]

    vocab_len = config["vocab_len"]

    language_model = GPT(
        dictionary_length=vocab_len,
        seq_len=seq_len,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_checkpointing=use_checkpointing,
        use_flash=use_flash,
    )

    language_model.load_state_dict(load_state_dict_strip_module(ckpt_path, map_location="cpu"), strict=True)
    language_model = language_model.to(device)
    language_model.eval()

    encoding = tiktoken.get_encoding("gpt2")
    test_prompt = "Alex went to vernal falls"
    generation_length = 40

    if global_rank == 0:
        with torch.no_grad():
            for temperature in [0.6, 0.7, 0.8, 0.9]:
                for k in [10, 20, 30]:
                    result = encoding.encode(text)
                    print(f'####################\nGenerating with temperature {temperature}, and k {k}\n-------------------')

                    for i in range(generation_length):
                        if len(result) > seq_len:
                            result_tensor = torch.tensor([result[-seq_len:]]).to(device)
                        else:
                            result_tensor = torch.tensor([result]).to(device)

                        outputs = language_model(result_tensor)
                        next_token = utils.top_k_sampling(outputs, k=k, temperature=temperature)
                        result.append(next_token.item())

                    print(f'{encoding.decode(result)}\n', flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("phase", type=str)
    parser.add_argument("config", type=str)

    args = parser.parse_args()

    setup()

    with open(args.config, "r") as f:
        config = json.load(f)

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")

    if args.phase == "train":
        run_train(config, device, local_rank, global_rank, world_size)

    else:
        run_test(config, device, local_rank, global_rank, world_size)

    cleanup()