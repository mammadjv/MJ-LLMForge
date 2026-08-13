import os
import argparse
import json
import sys
import time
import re
import string
from collections import Counter
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn import init as init
from torch.utils.data import DataLoader, DistributedSampler, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import RandomSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import bitsandbytes as bnb
import tiktoken
import matplotlib.pyplot as plt

import qa_dataset
from gpt import GPT
from transformer import utils


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def compute_em(pred, truth):
    return int(normalize_answer(pred) == normalize_answer(truth))


def compute_f1(pred, truth):
    pred_tokens = normalize_answer(pred).split()
    truth_tokens = normalize_answer(truth).split()

    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)

    return 2 * precision * recall / (precision + recall)


def load_state_dict_strip_module_qa(path, map_location="cpu"):
    state_dict = torch.load(path, map_location=map_location)

    if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        k = k.replace("module.", "")

        if "attention_mask" in k:
            continue

        new_state_dict[k] = v

    return new_state_dict

def load_state_dict_strip_module_gpt(path, map_location="cpu"):
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


def validate(val_dataloader, language_model, device, global_rank, vocab_len, world_size):
    language_model.eval()

    # Use float32 for accumulation
    total_loss = torch.zeros(1, device=device, dtype=torch.float32)

    total_tokens = torch.zeros(1, device=device, dtype=torch.float32)

    # SUM instead of MEAN
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

    if global_rank == 0:
        print("")

    stop_val_step = 50

    with torch.no_grad():
        for step, (inputs, labels, mask_y, causal_mask) in enumerate(val_dataloader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask_y = mask_y.to(device, non_blocking=True)
            causal_mask = causal_mask.to(device, non_blocking=True).unsqueeze(1)

            labels = labels.masked_fill(mask_y == -1, -1)
            valid_tokens = (labels != -1).sum()

            if valid_tokens.item() > 0:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = language_model(inputs, causal_mask)
                    loss_sum = loss_fn(outputs.view(-1, vocab_len), labels.view(-1))

                total_loss += loss_sum.detach().float()
                total_tokens += valid_tokens.float()

                if global_rank == 0:
                    batch_avg = loss_sum.detach().float()/ valid_tokens.float()

                    print(
                        f"\rPerforming validation. "
                        f"Iteration {step}: "
                        f"Loss {batch_avg.item():.4f}",
                        end="",
                        flush=True,
                    )

            if step >= stop_val_step:
                break

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
    avg_loss = total_loss / total_tokens.clamp_min(1)

    return avg_loss.item()


def run_train(config, device, local_rank, global_rank, world_size):
    base_lr = config['base_lr']
    seq_len = config['seq_len']
    embed_dim = config['embed_dim']
    num_heads = config['num_heads']
    num_layers = config['num_layers']
    n_frozen_layers = config['n_frozen_layers']

    iter_batch_size = config['iter_batch_size']
    actual_gpt_batch_size = config['actual_gpt_batch_size']

    epochs = config['epochs']
    use_checkpointing = config['use_checkpointing']
    use_flash = config['use_flash']
    max_steps = config['max_steps']
    max_warmup_steps = config['max_warmup_steps']
    val_step = config['val_step']
    lr_reduction_step = config['lr_reduction_step']
    num_workers = config['num_workers']
    data_dir = config['data_dir']
    gpt_path = config['gpt_path']

    language_model = GPT(
        dictionary_length=vocab_len,
        seq_len=seq_len,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_checkpointing=use_checkpointing,
        use_flash=use_flash,
    )

    grad_accum_steps = int(actual_gpt_batch_size/(seq_len * iter_batch_size * world_size)) #(2**12)/((2**10)*1*gpu_cores)
    qa_train_dataset = qa_dataset.QADataset(data_dir, 'train', iter_batch_size, seq_len, local_rank, world_size)
    qa_val_dataset = qa_dataset.QADataset(data_dir, 'val', iter_batch_size, seq_len, local_rank, world_size)

    train_dataloader = DataLoader(
        qa_train_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    
    val_dataloader = DataLoader(
        qa_val_dataset,
        batch_size=None,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )

    language_model.load_state_dict(load_state_dict_strip_module_gpt(gpt_path, map_location="cpu"), strict=True)

    for layer in language_model.transformer.layers[:n_frozen_layers]:
        for param in layer.parameters():
            param.requires_grad = False

    language_model = language_model.to(device)
    language_model = DDP(
        language_model,
        device_ids=[local_rank],
        broadcast_buffers=False,
    )

    decay, no_decay = utils.get_regularization_parameters(language_model, num_layers)

    decay = [p for p in decay if p.requires_grad]
    no_decay = [p for p in no_decay if p.requires_grad]

    optimizer = bnb.optim.AdamW8bit([
        {'params': decay, 'weight_decay': 0.1},
        {'params': no_decay, 'weight_decay': 0.0},
    ], lr=base_lr)

    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
    scaler = torch.amp.GradScaler()

    step = 0
    best_val_loss = float('inf')
    total_train_loss = torch.zeros(1, device=device, dtype=torch.float32)
    loss_accum_steps = torch.zeros(1, device=device, dtype=torch.float32)
    valid_batch_count = torch.zeros(1, device=device, dtype=torch.float32)  # Track valid batches locally

    optimizer.zero_grad()
    optimizer = utils.optimizer_warmup(optimizer, base_lr, 1, max_warmup_steps)
    first_clock = time.time()
    while True:
        if step/grad_accum_steps > max_steps:
            break

        for batch_idx, (inputs, labels, mask_y, causal_mask) in enumerate(train_dataloader):
            step += 1
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask_y = mask_y.to(device, non_blocking=True)
            labels[mask_y == -1] = -1

            causal_mask = causal_mask.to(device, non_blocking=True).unsqueeze(1)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                is_sync_step = (step % grad_accum_steps == 0)
                language_model.require_backward_grad_sync = is_sync_step
                outputs = language_model(inputs, causal_mask)
                loss = loss_fn(outputs.view(-1, vocab_len), labels.view(-1))

                if torch.isnan(loss):
                    loss = outputs.sum() * 0.0

                else:
                    loss_accum_steps += loss.detach()
                    valid_batch_count += 1.0

                scaled_loss = loss / grad_accum_steps
                scaler.scale(scaled_loss).backward()

            if is_sync_step:
                scaler.unscale_(optimizer)  # must unscale before clipping
                torch.nn.utils.clip_grad_norm_(language_model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                # All-reduce losses and valid counts across all ranks
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
                    optimizer = utils.cosine_lr_update(optimizer, update_step, base_lr, max_steps, max_warmup_steps, min_lr=6e-5)

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
                        torch.save(checkpoint, './best_model_qa_all_{}.pt'.format(n_frozen_layers))

                    total_train_loss.zero_()
                    dist.barrier(device_ids=[local_rank])
                    language_model.train()


def run_test(config, device, local_rank):
    base_lr = config['base_lr']
    seq_len = config['seq_len']
    embed_dim = config['embed_dim']
    num_heads = config['num_heads']
    num_layers = config['num_layers']
    n_frozen_layers = config['n_frozen_layers']

    iter_batch_size = config['iter_batch_size']
    actual_gpt_batch_size = config['actual_gpt_batch_size']

    epochs = config['epochs']
    use_checkpointing = config['use_checkpointing']
    use_flash = config['use_flash']
    max_steps = config['max_steps']
    max_warmup_steps = config['max_warmup_steps']
    val_step = config['val_step']
    lr_reduction_step = config['lr_reduction_step']
    num_workers = config['num_workers']
    data_dir = config['data_dir']
    gpt_path = config['gpt_path']

    language_model = GPT(
        dictionary_length=vocab_len,
        seq_len=seq_len,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_checkpointing=use_checkpointing,
        use_flash=use_flash,
    )

    language_model = language_model.to(device)  # separate line, on the model
    language_model = DDP(language_model,
                         device_ids=[local_rank],
                         output_device=local_rank,
                         broadcast_buffers=False,
                        )

    checkpoints = [
        './best_model_qa_all_0.pt',
        './best_model_qa_all_4.pt',
    ]
    for ckpt_path in checkpoints:
        print("\n" + "=" * 60)
        print("Testing checkpoint:", ckpt_path)
        print("=" * 60)
        language_model.module.load_state_dict(load_state_dict_strip_module_qa(ckpt_path, map_location="cpu"), strict=False)
        language_model.eval()
        test_dataset = qa_dataset.QADataset(data_dir=data_dir, split='test', batch_size=1, seq_len=seq_len, rank=0, world_size=1)
        test_loader = DataLoader(test_dataset, batch_size=1)
        eot_token = test_dataset.word_encoding.eot_token

        with torch.no_grad():
            total_em = 0
            total_f1 = 0
            num_examples = 0
            for idx, (prompt, ground_truth) in enumerate(test_loader):
                x = prompt.to(device=device, dtype=torch.long)
                prompt_len = x.size(1)

                for _ in range(150):
                    curr_T = x.size(1)
                    causal = torch.tril(torch.ones(curr_T, curr_T, device=device, dtype=torch.bool))
                    attn_mask = torch.where(causal, 0.0, float("-inf")).unsqueeze(0).unsqueeze(1)
                    logits = language_model(x, attn_mask)

                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    x = torch.cat([x, next_token], dim=1)

                    if next_token.item() == eot_token:
                        break

                generated_ids = x[0, prompt_len:].cpu().tolist()
                generated_text = test_dataset.word_encoding.decode(generated_ids)

                # Ground-truth answer
                ground_truth_ids = ground_truth[0].cpu().tolist()
                ground_truth_text = test_dataset.word_encoding.decode(ground_truth_ids)

                generated_clean = generated_text.replace("<|endoftext|>", "")
                ground_truth_clean = ground_truth_text.replace("<|endoftext|>", "")

                em = compute_em(generated_clean, ground_truth_clean)
                f1 = compute_f1(generated_clean, ground_truth_clean)

                total_em += em
                total_f1 += f1
                num_examples += 1

                if idx % 1000 == 0:
                    print(test_dataset.word_encoding.decode(x[0].cpu().tolist()))
                    print("Generated answer:{}".format(generated_text), flush=True)
                    print("Ground truth:{}".format(ground_truth_text), flush=True)
                    print("EM: {}, F1: {:.4f}".format(em, f1), flush=True)
                    print("-" * 30, flush=True)

            avg_em = 100 * total_em / num_examples
            avg_f1 = 100 * total_f1 / num_examples

            print("\nCheckpoint:", ckpt_path)
            print("EM: {:.2f}%".format(avg_em))
            print("F1: {:.2f}%".format(avg_f1))


if __name__ == "__main__":
    setup()

    phase = str(sys.argv[1])
    config = str(sys.argv[2])

    with open("config.json", "r") as f:
        config = json.load(f)

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    device = torch.device(f'cuda:{local_rank}')
    vocab_len = qa_dataset.get_vocab_len()

    if phase == 'train':
        run_train(config, device, local_rank, global_rank, world_size)

    else:
        run_test(config, device, local_rank)

    cleanup()