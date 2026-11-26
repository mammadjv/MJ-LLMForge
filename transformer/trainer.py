import os
import time
from collections import OrderedDict
import random

import numpy as np
import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions
)

from torch.nn.parallel import DistributedDataParallel as DDP
from transformer import utils



def get_rng_state(device):
    np_state = np.random.get_state()

    return {
        "python": random.getstate(),
        "numpy": (
            np_state[0],
            np_state[1].tolist(),
            np_state[2],
            np_state[3],
            np_state[4],
        ),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
    }


def restore_rng_state(state, device):
    random.setstate(state["python"])

    n = state["numpy"]
    np.random.set_state((
        n[0],
        np.asarray(n[1], dtype=np.uint32),
        n[2],
        n[3],
        n[4],
    ))

    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(
        state["torch_cuda"],
        device=device,
    )


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


class Trainer:
    def __init__(self,
                 language_model,
                 loss_fn,
                 device,
                 global_rank,
                 world_size,
                 local_rank,
                 config,
                 trainer_mode='ddp',
                ):
        self.language_model = language_model
        self.loss_fn = loss_fn
        self.scaler = torch.amp.GradScaler()
        self.global_rank=global_rank
        self.world_size=world_size
        self.device = device
        self.local_rank=local_rank
        self.trainer_mode = trainer_mode
        self.train_dataloader = None
        self.val_dataloader = None
        self.config = config
        self._resume_rng_state = None

        self.load_config()

        if self.trainer_mode == 'fsdp2':
            self.configure_trainable_parameters()

            for layer in self.language_model.transformer.layers:
                fully_shard(layer)

            fully_shard(self.language_model)

            self.create_optimizer()
            self.load_checkpoint()

        else:
            self.configure_trainable_parameters()
            self.load_checkpoint()
            if self.config['lora_rank'] > 0 and self.config['qlora'] == True:
                utils.replace_with_4bit(self.language_model.transformer)

            self.language_model.to(self.local_rank)
            self.language_model = DDP(
                    self.language_model,
                    device_ids=[self.local_rank],
                    broadcast_buffers=False,
                )

            self.create_optimizer()
            if self.optimizer_state_dict is not None:
                self.optimizer.load_state_dict(self.optimizer_state_dict)



    def update_dataloaders(self,
                          train_dataloader,
                          val_dataloader,
                          ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader


    def create_optimizer(self):
        decay, no_decay = utils.get_regularization_parameters(self.language_model, self.config['num_layers'])

        if self.trainer_mode == 'ddp':
            self.optimizer = bnb.optim.AdamW8bit([
                {'params': decay, 'weight_decay': 0.1},
                {'params': no_decay, 'weight_decay': 0.0}
            ], lr=self.base_lr, betas=(0.9, 0.95))

        else:
            self.optimizer = torch.optim.AdamW([
                {'params': decay, 'weight_decay': 0.1},
                {'params': no_decay, 'weight_decay': 0.0}
            ], lr=self.base_lr, betas=(0.9, 0.95))

        self.optimizer = utils.optimizer_warmup(self.optimizer, self.base_lr, 1, self.max_warmup_steps)


    def configure_trainable_parameters(self):
        # finetune or not
        if self.last_ckpt_path != "":
            if self.config['lora_rank'] > 0:
                for param in self.language_model.parameters():
                    param.requires_grad = False

                for name, param in self.language_model.named_parameters():
                    if any(x in name for x in ["Aq", "Bq", "Ak", "Bk", "Av", "Bv"]):
                        param.requires_grad = True

            else:
                if self.config["n_frozen_layers"] > 0:
                    self.language_model.embeddin_table.weight.requires_grad_(False)
                    self.language_model.pos_embed.weight.requires_grad_(False)

                for layer in self.language_model.transformer.layers[:self.config['n_frozen_layers']]:
                    for param in layer.parameters():
                        param.requires_grad = False


    def load_config(self):
        self.max_steps = self.config['max_steps']
        self.base_lr = self.config['base_lr']
        self.vocab_len = self.config['vocab_len']
        self.seq_len = self.config['seq_len']
        self.iter_batch_size = self.config['iter_batch_size']
        self.min_lr = self.config['min_lr']
        self.max_warmup_steps = self.config['max_warmup_steps']
        self.val_step = self.config['val_step']
        self.iter_batch_size = self.config["iter_batch_size"]
        self.actual_gpt_batch_size = self.config["actual_gpt_batch_size"]
        self.grad_accum_steps = int(self.actual_gpt_batch_size/(self.seq_len * self.iter_batch_size * self.world_size))
        self.best_ckpt_path = self.config['best_ckpt_path']
        self.current_ckpt_path = self.config['current_ckpt_path']
        self.last_ckpt_path = self.config['last_ckpt_path']
        self.lr_cosine_steps = self.config['lr_cosine_steps']
        self.qlora = self.config['qlora']

        if self.config['lora_rank'] == 0 and self.config['n_frozen_layers'] > 0:
            suffix = f"_frozen_{self.config['n_frozen_layers']}"
            root, ext = os.path.splitext(self.current_ckpt_path)
            self.current_ckpt_path = f"{root}{suffix}{ext}"
            root, ext = os.path.splitext(self.best_ckpt_path)
            self.best_ckpt_path = f"{root}{suffix}{ext}"


    def load_checkpoint(self):
        self.step = 0
        self.update_step = 0
        self.best_val_loss = float('inf')
        self.optimizer_state_dict = None
        
        if self.last_ckpt_path != "":
            if self.trainer_mode == 'fsdp2':
                if self.global_rank == 0:
                    checkpoint = torch.load(self.last_ckpt_path, map_location="cpu")
                    model_state_dict = checkpoint["model_state_dict"]
    
                    if self.config['resume']:
                        optimizer_state_dict = checkpoint["optimizer_state_dict"]
                    else:
                        optimizer_state_dict = {}
    
                else:
                    checkpoint = None
                    model_state_dict = {}
                    optimizer_state_dict = {}
    
                options = StateDictOptions(
                    full_state_dict=True,
                    broadcast_from_rank0=True,
                    strict=False,
                )
    
                set_model_state_dict(
                    self.language_model,
                    model_state_dict,
                    options=options
                )
    
                if self.config['resume']:
                    set_optimizer_state_dict(
                        self.language_model,
                        self.optimizer,
                        optimizer_state_dict,
                        options=options
                    )
    
                    if self.global_rank == 0:
                        metadata = {
                            "step": checkpoint["step"],
                            "update_step": checkpoint["update_step"],
                            "best_val_loss": checkpoint["best_val_loss"],
                            "scaler_state_dict": checkpoint["scaler_state_dict"],
                            "rng_states_by_rank": checkpoint.get(
                                "rng_states_by_rank"
                            ),
                            "world_size": checkpoint.get("world_size"),
                        }
    
                    else:
                        metadata = None
    
                    metadata_list = [metadata]
                    dist.broadcast_object_list(metadata_list, src=0)
                    metadata = metadata_list[0]
    
                    self.step = metadata["step"]
                    self.update_step = metadata["update_step"]
                    self.best_val_loss = metadata["best_val_loss"]
                    self.scaler.load_state_dict(metadata["scaler_state_dict"])
                    if metadata.get("rng_states_by_rank") is not None:
                        assert metadata.get("world_size", self.world_size) == self.world_size
                    
                        self._resume_rng_state = metadata[
                            "rng_states_by_rank"
                        ][self.global_rank]
    
            

            else:  # DDP, fresh fine-tuning from a pretrained backbone
                if self.global_rank == 0 or self.qlora:
                    print("Loading pretrained GPT checkpoint on rank 0...", flush=True)
            
                    checkpoint = torch.load(
                        self.last_ckpt_path,
                        map_location="cpu",
                        mmap=True,
                        weights_only=False,  # Only for checkpoints you trust
                    )
            
                    state = checkpoint.get("model_state_dict", checkpoint)
            
                    state = {
                        k.removeprefix("module."): v
                        for k, v in state.items()
                        if "attention_mask" not in k
                    }
            
                    result = self.language_model.load_state_dict(
                        state,
                        strict=False,
                    )
            
                    adapter_names = {"Aq", "Bq", "Ak", "Bk", "Av", "Bv"}
            
                    expected_missing = {
                        name
                        for name, _ in self.language_model.named_parameters()
                        if name.rsplit(".", 1)[-1] in adapter_names
                    }
            
                    if (
                        set(result.missing_keys) != expected_missing
                        or result.unexpected_keys
                    ):
                        raise RuntimeError(
                            f"Checkpoint mismatch: "
                            f"missing={result.missing_keys}, "
                            f"unexpected={result.unexpected_keys}"
                        )
            
                    del state, checkpoint
                    print("Pretrained weights loaded successfully.", flush=True)
            
                # No manual broadcast needed here.
                # The subsequent DDP constructor synchronizes model parameters
                # from rank 0 to all other ranks.




    def validate(self):
        self.language_model.eval()
    
        total_loss = torch.zeros(1, device=self.device, dtype=torch.float32)
        count = torch.zeros(1, device=self.device, dtype=torch.float32)
    
        if self.global_rank == 0:
            print('')
    
        step = 0
        stop_val_step = 50
        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(self.val_dataloader):
                step += 1
                inputs, labels = self.prepare_inputs(inputs, labels)
    
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    outputs = self.language_model(*inputs)
                    loss = self.loss_fn(outputs.view(-1, self.vocab_len), labels.view(-1))
    
                total_loss += loss.detach().to(total_loss.dtype)
                count += 1.0
    
                if self.global_rank == 0:
                    print(f"\rPerforming validation. Iteration {step}: Loss {loss.item()}", end="", flush=True)
    
                if step % stop_val_step == 0:
                    break
    
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        avg = total_loss / count
    
        return float(avg.item())

    def prepare_inputs(self, inputs, labels):
        inputs = tuple(item.to(self.device, non_blocking=True) for item in inputs)
        labels = labels.to(self.device, non_blocking=True)
        return inputs, labels
    
    def start(self):
        if self.train_dataloader is None or self.val_dataloader is None:
            raise RuntimeError('Train and validation DataLoaders need to be initialized.')

        total_train_loss = torch.zeros(1, device=self.device, dtype=torch.float32)
        loss_accum_steps = torch.zeros(1, device=self.device, dtype=torch.float32)
        valid_batch_count = torch.zeros(1, device=self.device, dtype=torch.float32)

        self.optimizer.zero_grad()
        first_clock = time.time()
        if self.global_rank == 0:
            print('Start of Training', flush=True)

        debug = False
        while True:
            if self.step // self.grad_accum_steps >= self.max_steps:
                break
        
            for batch_idx, (inputs, labels) in enumerate(self.train_dataloader):
                self.step += 1
                inputs, labels = self.prepare_inputs(inputs, labels)
        
                if self._resume_rng_state is not None:
                    restore_rng_state(
                        self._resume_rng_state,
                        self.device,
                    )
                    self._resume_rng_state = None

                is_sync_step = (self.step % self.grad_accum_steps == 0)
                current_update = self.step // self.grad_accum_steps
        
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    if self.trainer_mode == 'fsdp2':
                        self.language_model.set_requires_gradient_sync(is_sync_step)
                        self.language_model.set_is_last_backward(is_sync_step)

                    else:
                        self.language_model.require_backward_grad_sync = is_sync_step
        
                    outputs = self.language_model(*inputs)
                    loss = self.loss_fn(outputs.view(-1, self.vocab_len), labels.view(-1))

                    if not torch.isfinite(loss).item():
                        raise FloatingPointError(
                            f"Nonfinite training loss at microbatch "
                            f"{self.step}, rank {self.global_rank}"
                        )

                    else:
                        loss_accum_steps += loss.detach()
                        valid_batch_count += 1.0

                    scaled_loss = loss / self.grad_accum_steps
                    self.scaler.scale(scaled_loss).backward()

                if is_sync_step:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.language_model.parameters(), max_norm=0.5)

                    if self.global_rank == 0:
                        grad_norm_item = grad_norm.item()
                        loss_item = loss.item()
                        lr = self.optimizer.param_groups[0]['lr']
                        old_scale = self.scaler.get_scale()

                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    if self.global_rank == 0:
                        new_scale = self.scaler.get_scale()

                    dist.all_reduce(loss_accum_steps, op=dist.ReduceOp.SUM)
                    dist.all_reduce(valid_batch_count, op=dist.ReduceOp.SUM)
        
                    valid_batch_count = torch.clamp(valid_batch_count, min=1.0)
                    step_avg_loss = loss_accum_steps / valid_batch_count
                    total_train_loss += step_avg_loss
        
                    self.update_step = self.step // self.grad_accum_steps
        
                    if self.global_rank == 0:
                        elapsed_time = time.time() - first_clock
                        tokens_per_sec = (self.grad_accum_steps * self.iter_batch_size * self.seq_len * self.world_size) / elapsed_time
                
                        print(
                            f"\rStep {self.update_step}: "
                            f"Average Loss {step_avg_loss.item():.4f}, "
                            f"loss={loss_item:.4f}, "
                            f"grad_norm={grad_norm_item:.4f}, "
                            f"lr={lr:.8e}, "
                            f"scale={old_scale}->{new_scale}, "
                            f"tokens/sec={tokens_per_sec:.2f}",
                            end="", flush=True
                        )

                        first_clock = time.time()

                    loss_accum_steps.zero_()
                    valid_batch_count.zero_()
                    self.optimizer.zero_grad()

                    if self.update_step <= self.max_warmup_steps:
                        self.optimizer = utils.optimizer_warmup(
                            self.optimizer,
                            self.base_lr,
                            self.update_step,
                            self.max_warmup_steps,
                        )
                    else:
                        self.optimizer = utils.cosine_lr_update(
                            self.optimizer,
                            self.update_step,
                            self.base_lr,
                            self.lr_cosine_steps,
                            self.max_warmup_steps,
                            min_lr=self.min_lr,
                        )
        
                    if self.update_step % self.val_step == 0:
                        val_loss = self.validate()
        
                        dist.all_reduce(total_train_loss, op=dist.ReduceOp.SUM)
                        total_train_loss /= (self.world_size * self.val_step)
        
                        is_best = val_loss < self.best_val_loss
                        self.best_val_loss = min(self.best_val_loss, val_loss)

                        if self.global_rank == 0:
                            print(f'\n Step {self.update_step}: avg train loss {total_train_loss.item()}, avg val loss {val_loss}', flush=True)

                        if self.trainer_mode == 'fsdp2':
                            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
                            model_state_dict = get_model_state_dict(self.language_model, options=options)
                            optimizer_state_dict = get_optimizer_state_dict(self.language_model, self.optimizer, options=options)

                        elif self.global_rank == 0:
                            model_state_dict = self.language_model.module.state_dict()
                            optimizer_state_dict = self.optimizer.state_dict()

                        # Every rank participates
                        local_rng = get_rng_state(self.device)
                        
                        rng_states_by_rank = (
                            [None] * self.world_size
                            if self.global_rank == 0
                            else None
                        )
                        
                        dist.gather_object(
                            local_rng,
                            object_gather_list=rng_states_by_rank,
                            dst=0,
                        )
                        if self.global_rank == 0:
                            if self.config['lora_rank'] == 0:
                                checkpoint = {
                                    "model_state_dict": model_state_dict,
                                    "optimizer_state_dict": optimizer_state_dict,
                                    "scaler_state_dict": self.scaler.state_dict(),
                                
                                    "step": self.step,
                                    "update_step": self.update_step,
                                    "best_val_loss": self.best_val_loss,
                                
                                    # Reproducibility
                                    "rng_states_by_rank": rng_states_by_rank,
                                    "config": dict(self.config),
                                    "world_size": self.world_size,
                                    "torch_version": str(torch.__version__),
                                }

                                temp_path = self.current_ckpt_path + ".tmp"
                                torch.save(checkpoint, temp_path)
                                os.replace(temp_path, self.current_ckpt_path)

                                if is_best:
                                    best_temp = self.best_ckpt_path + ".tmp"
                                    torch.save(checkpoint, best_temp)
                                    os.replace(best_temp, self.best_ckpt_path)

                            elif is_best:
                                lora_state_dict = {
                                    name: param.detach().cpu()
                                    for name, param in model_state_dict.items()
                                    if any(key in name for key in ["Aq", "Bq", "Ak", "Bk", "Av", "Bv"])
                                }

                                checkpoint = {
                                    'model_state_dict': lora_state_dict,
                                    'optimizer_state_dict': optimizer_state_dict,
                                    'scaler_state_dict': self.scaler.state_dict(),
                                    'step': self.step,
                                    'update_step': self.update_step,
                                    'best_val_loss': self.best_val_loss,
                                    }

                                lora_prefix = "" if not self.qlora else "q"
                                torch.save(
                                    checkpoint,
                                    f"{lora_prefix}lora_weights_{self.config['lora_rank']}.pt"
                                )

                        total_train_loss.zero_()
                        dist.barrier(device_ids=[self.local_rank])
                        self.language_model.train()
        
                    if self.update_step >= self.max_steps:
                        break


