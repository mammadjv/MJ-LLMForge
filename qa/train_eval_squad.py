import os
import argparse
import json
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn import init as init
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.data import RandomSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import bitsandbytes as bnb
import tiktoken
from collections import OrderedDict
import matplotlib.pyplot as plt

import qa_dataset
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
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    if global_rank == 0:
        print('')
    
    step = 0
    stop_val_step = 50
    with torch.no_grad():
        for batch_idx, (inputs, labels, mask_y, causal_mask) in enumerate(val_dataloader):
            step += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels[mask_y == -1] = -100
            causal_mask = causal_mask.to(device, non_blocking=True).unsqueeze(1)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = language_model(inputs, causal_mask)
                loss = loss_fn(outputs.view(-1, vocab_len), labels.view(-1))

            total_loss += loss.detach().to(total_loss.dtype)
            count += 1.0

            if global_rank == 0:
                print(f"\rPerforming validation. Iteration {step}: Loss {loss.item()}", end="", flush=True)

            if step% stop_val_step == 0:
                break                

    # Global reduce (SUM)
    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    avg = total_loss / count

    return float(avg.item())


if __name__ == "__main__":
    setup()

    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    device = torch.device(f'cuda:{local_rank}')

    base_lr = 6e-5
    seq_len = 2**10
    embed_dim = 1024
    num_heads=16
    num_layers = 24

    iter_batch_size = 4
    actual_gpt_batch_size = 2**17 #0.5M
    grad_accum_steps = int(actual_gpt_batch_size/(seq_len * iter_batch_size * world_size)) #(2**19)/((2**10)*4*gpu_cores)
    max_warmup_steps = 1
    val_step = 200
    lr_reduction_step = 400

    epochs = 20
    use_checkpointing = False
    use_flash=True
    max_steps = 600000

    phase = str(sys.argv[1])
    
    data_dir = '../dataset/squadv2'
    qa_train_dataset = qa_dataset.QADataset(data_dir, 'train', iter_batch_size, seq_len, local_rank, world_size)
    qa_val_dataset = qa_dataset.QADataset(data_dir, 'val', iter_batch_size, seq_len, local_rank, world_size)

    vocab_len = qa_train_dataset.vocab_len

    train_dataloader = DataLoader(
        qa_train_dataset,
        batch_size=None,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True
    )
    
    val_dataloader = DataLoader(
        qa_val_dataset,
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

    global_rank = dist.get_rank()

    if phase == 'train':
        ckpt_path = './best_model.pt'
        language_model.load_state_dict(load_state_dict_strip_module(ckpt_path, map_location="cpu"), strict=True)
        language_model = language_model.to(device)
        for layer in language_model.transformer.layers[:-8]:
            for param in layer.parameters():
                param.requires_grad = False

        language_model = DDP(language_model, device_ids=[local_rank])

        decay, no_decay = utils.get_regularization_parameters(language_model, num_layers)

        decay = [p for p in decay if p.requires_grad]
        no_decay = [p for p in no_decay if p.requires_grad]
    
        optimizer = bnb.optim.AdamW8bit([
            {'params': decay, 'weight_decay': 0.1},
            {'params': no_decay, 'weight_decay': 0.0},
        ], lr=base_lr)
    
        loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        scaler = torch.amp.GradScaler()
        language_model.apply(utils.weights_init)
        step = 0
        best_val_loss = float('inf')
        total_train_loss = torch.zeros(1, device=device, dtype=torch.float16)
        loss_accum_steps = torch.zeros(1, device=device, dtype=torch.float32)

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
                labels[mask_y == -1] = -1

                causal_mask = causal_mask.to(device, non_blocking=True).unsqueeze(1)

                # for gradient_accumulation
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = language_model(inputs, causal_mask)
                    loss = loss_fn(outputs.view(-1, vocab_len), labels.view(-1))

                loss_accum_steps += loss.detach()/grad_accum_steps

                scaled_loss = loss / grad_accum_steps

                if step % grad_accum_steps != 0:
                    with language_model.no_sync():
                        scaler.scale(scaled_loss).backward()

                # every replica accumulated steps
                else:
                    scaler.scale(scaled_loss).backward()
                    scaler.unscale_(optimizer)  # must unscale before clipping
                    torch.nn.utils.clip_grad_norm_(language_model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()

                    total_train_loss += loss_accum_steps

                    update_step = step // grad_accum_steps
                    if global_rank == 0:
                        elapsed_time = time.time() - first_clock
                        tokens_per_sec = (grad_accum_steps * iter_batch_size * seq_len * world_size) / elapsed_time
                        elapsed_time = '{:.3f}'.format(time.time() - first_clock)
                        print(f"\rStep {update_step}: Loss {loss_accum_steps.item()}, batch time {tokens_per_sec}, grad accum steps {grad_accum_steps}", end="", flush=True)
                        first_clock = time.time()

                    loss_accum_steps.zero_()
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
                        total_train_loss /= (world_size*val_step)
    
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
                            torch.save(checkpoint, f'./best_model1.pt')
    
                        total_train_loss.zero_()
                        dist.barrier(device_ids=[local_rank])
                        language_model.train()


        cleanup()

    else:
        ckpt_path = './best_model.pt'
        language_model.load_state_dict(load_state_dict_strip_module(ckpt_path, map_location="cpu"), strict=True)
        language_model = language_model.to(device)  # separate line, on the model
        language_model = DDP(language_model, device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        encoding = tiktoken.get_encoding("gpt2")
        text = "Alex went to vernal falls"

        for temperature in [0.6, 0.7, 0.8, 0.9]:
            for k in [10, 20, 30]:
                result = encoding.encode(text)
                print(f'####################\nGenerating with temperature {temperature}, and k {k}\n-------------------')

                # set your desired text length
                for i in range(40):
                    if len(result) > seq_len:
                      result_tensor = torch.tensor([result[-seq_len:]]).to(device)

                    else:
                      result_tensor = torch.tensor([result]).to(device)

                    outputs = language_model(result_tensor)
                    next_token = utils.top_k_sampling(outputs, k=k, temperature=temperature)
                    result.append(next_token.item())
    
                # or save it wherever you want
                print(f'{encoding.decode(result)}\n<End of poem with 1000 characters>', flush=True)
