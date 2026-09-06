import os
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import DataLoader

from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict, set_model_state_dict, set_optimizer_state_dict, StateDictOptions

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
    num_workers = config['num_workers']

    gpt_train_dataset = gpt_dataset.GPTDataset(data_dir, 'train', iter_batch_size, seq_len, global_rank, world_size, num_workers)
    gpt_val_dataset = gpt_dataset.GPTDataset(data_dir, 'val', iter_batch_size, seq_len, global_rank, world_size, num_workers)

    vocab_len = gpt_train_dataset.vocab_len

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

    for layer in language_model.transformer.layers:
        fully_shard(layer)

    fully_shard(language_model)
    
    decay, no_decay = utils.get_regularization_parameters(language_model, num_layers)
    
    optimizer = torch.optim.AdamW([
        {'params': decay, 'weight_decay': 0.1},
        {'params': no_decay, 'weight_decay': 0.0},
    ], betas=(0.9, 0.95), lr=base_lr)

    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler()

    if config['resume'] == False:
        step = 0
        update_step = 0
        best_val_loss = float('inf')
        optimizer = utils.optimizer_warmup(optimizer, base_lr, 1, max_warmup_steps)

    else:
        if global_rank == 0:
            checkpoint = torch.load(ckpt_path, map_location="cpu")
            model_state_dict = checkpoint["model_state_dict"]
            optimizer_state_dict = checkpoint["optimizer_state_dict"]
        else:
            checkpoint = None
            model_state_dict = {}
            optimizer_state_dict = {}
    
        options = StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
        )
    
        set_model_state_dict(
            language_model,
            model_state_dict,
            options=options
        )
    
        set_optimizer_state_dict(
            language_model,
            optimizer,
            optimizer_state_dict,
            options=options
        )
    
        if global_rank == 0:
            metadata = {
                "step": checkpoint["step"],
                "update_step": checkpoint["update_step"],
                "best_val_loss": checkpoint["best_val_loss"],
                "scaler_state_dict": checkpoint["scaler_state_dict"],
            }
        else:
            metadata = None
    
        metadata_list = [metadata]
        dist.broadcast_object_list(metadata_list, src=0)
        metadata = metadata_list[0]
    
        step = metadata["step"]
        update_step = metadata["update_step"]
        best_val_loss = metadata["best_val_loss"]
        scaler.load_state_dict(metadata["scaler_state_dict"])
    
    gpt_train_dataset.update_dataset_sharding(update_step, grad_accum_steps)
    
    train_dataloader = DataLoader(
        gpt_train_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    val_dataloader = DataLoader(
        gpt_val_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    for group in optimizer.param_groups:
        beta1, beta2 = group["betas"]
        group["betas"] = (beta1, 0.95)

    total_train_loss = torch.zeros(1, device=device, dtype=torch.float16)
    loss_accum_steps = torch.zeros(1, device=device, dtype=torch.float32)
    valid_batch_count = torch.zeros(1, device=device, dtype=torch.float32)

    optimizer.zero_grad()
    first_clock = time.time()
    print('Start of Training', flush=True)
    
    while True:
        if step // grad_accum_steps >= max_steps:
            break
    
        for batch_idx, (inputs, labels) in enumerate(train_dataloader):
            step += 1
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
    
            is_sync_step = (step % grad_accum_steps == 0)
            current_update = step // grad_accum_steps
            # debug = 5895 <= current_update <= 6500
            debug = False
    
            with torch.amp.autocast("cuda", dtype=torch.float16):
                language_model.set_requires_gradient_sync(is_sync_step)
    
                outputs = language_model(inputs)
    
                # Small sampled logit inspection, only once per optimizer update
                if is_sync_step and debug and global_rank == 0:
                    with torch.no_grad():
                        logits_debug = outputs.detach()[0, :16, :1024].float()
                        logits_absmax = logits_debug.abs().max().item()
                        logits_mean = logits_debug.mean().item()
                        logits_std = logits_debug.std().item()
    
                loss = loss_fn(outputs.view(-1, vocab_len), labels.view(-1))
    
                loss_accum_steps += loss.detach()
                valid_batch_count += 1.0
    
                scaled_loss = loss / grad_accum_steps
                scaler.scale(scaled_loss).backward()
    
            if is_sync_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(language_model.parameters(), max_norm=1.0)
    
                if global_rank == 0:
                    grad_norm_item = grad_norm.item()
                    loss_item = loss.item()
                    lr = optimizer.param_groups[0]['lr']
                    old_scale = scaler.get_scale()
    
                # Track one parameter only during debug window
                debug_param = None
                debug_name = None
                before = None
                exp_avg_norm_item = None
                exp_avg_sq_norm_item = None
                grad_after_clip_norm_item = None

                v_mean_item = float("nan")
                v_min_item = float("nan")
                v_max_item = float("nan")
                adam_direction_norm_item = float("nan")
                
                if debug:
                    for name, p in language_model.named_parameters():
                        if name == "transformer.layers.0.k.weight":
                            debug_name = name
                            debug_param = p
                            before = p.detach().clone()
                
                            if p.grad is not None:
                                grad_after_clip_norm_item = p.grad.detach().float().norm().item()
                
                            state = optimizer.state.get(p, None)
                
                            if state is not None and len(state) > 0:
                                exp_avg = state.get("exp_avg", None)
                                exp_avg_sq = state.get("exp_avg_sq", None)
                
                                if exp_avg is not None and exp_avg_sq is not None:
                                    m = exp_avg.detach().float()
                                    v = exp_avg_sq.detach().float()
                                    eps = optimizer.param_groups[0]["eps"]
                
                                    exp_avg_norm_item = m.norm().item()
                                    exp_avg_sq_norm_item = v.norm().item()
                                    v_mean_item = v.mean().item()
                                    v_min_item = v.min().item()
                                    v_max_item = v.max().item()
                                    adam_direction_norm_item = (m / (v.sqrt() + eps)).norm().item()
                
                            break
    
                scaler.step(optimizer)
                scaler.update()
    
                if global_rank == 0:
                    new_scale = scaler.get_scale()
    
                if debug and debug_param is not None:
                    update_norm = (debug_param.detach() - before).float().norm()
                    param_norm = debug_param.detach().float().norm()
            
                    if global_rank == 0:
                        update_norm_item = update_norm.item()
                        param_norm_item = param_norm.item()
    
                dist.all_reduce(loss_accum_steps, op=dist.ReduceOp.SUM)
                dist.all_reduce(valid_batch_count, op=dist.ReduceOp.SUM)
    
                valid_batch_count = torch.clamp(valid_batch_count, min=1.0)
                step_avg_loss = loss_accum_steps / valid_batch_count
                total_train_loss += step_avg_loss
    
                update_step = step // grad_accum_steps
    
                if global_rank == 0:
                    elapsed_time = time.time() - first_clock
                    tokens_per_sec = (grad_accum_steps * iter_batch_size * seq_len * world_size) / elapsed_time
            
                    print(
                        f"\rStep {update_step}: "
                        f"Average Loss {step_avg_loss.item():.4f}, "
                        f"loss={loss_item:.4f}, "
                        f"grad_norm={grad_norm_item:.4f}, "
                        f"lr={lr:.8e}, "
                        f"scale={old_scale}->{new_scale}, "
                        f"tokens/sec={tokens_per_sec:.2f}",
                        end="", flush=True
                    )
            
                    if debug and debug_param is not None:
                        print(
                            f"\nDEBUG update={update_step} "
                            f"param={debug_name} "
                            f"param_norm={param_norm_item:.6f} "
                            f"update_norm={update_norm_item:.8f} "
                            f"grad_after_clip={grad_after_clip_norm_item:.8e} "
                            f"exp_avg_norm={exp_avg_norm_item:.8e} "
                            f"exp_avg_sq_norm={exp_avg_sq_norm_item:.8e} "
                            f"v_mean={v_mean_item:.8e} "
                            f"v_min={v_min_item:.8e} "
                            f"v_max={v_max_item:.8e} "
                            f"adam_direction_norm={adam_direction_norm_item:.8e}",
                            flush=True
                        )
            
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
    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
    
                        options = StateDictOptions(
                            full_state_dict=True,
                            cpu_offload=True,
                        )
    
                        model_state_dict = get_model_state_dict(
                            language_model,
                            options=options
                        )
    
                        optimizer_state_dict = get_optimizer_state_dict(
                            language_model,
                            optimizer,
                            options=options
                        )
    
                        if global_rank == 0:
                            checkpoint = {
                                'model_state_dict': model_state_dict,
                                'optimizer_state_dict': optimizer_state_dict,
                                'scaler_state_dict': scaler.state_dict(),
                                'step': step,
                                'update_step': update_step,
                                'best_val_loss': best_val_loss,
                            }
                            torch.save(checkpoint, ckpt_path)
    
                    total_train_loss.zero_()
                    dist.barrier(device_ids=[local_rank])
                    language_model.train()
    
                if update_step >= max_steps:
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
        use_checkpointing=False,
        use_flash=use_flash,
    )

    language_model.load_state_dict(load_state_dict_strip_module(ckpt_path, map_location="cpu"), strict=True)
    language_model = language_model.to(device)
    language_model.eval()

    encoding = tiktoken.get_encoding("gpt2")
    text = "The dog walked into the kitchen"
    generation_length = 150

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