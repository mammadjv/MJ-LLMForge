import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions
)

from transformer import utils

class Trainer:
    def __init__(self,
                 language_model,
                 optimizer,
                 loss_fn,
                 device,
                 global_rank,
                 world_size,
                 local_rank,
                 config,
                 trainer_mode='ddp',
                ):
        self.language_model = language_model
        self.optimizer = optimizer
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
        self.load_config()
        self.load_checkpoint()


    def update_dataloaders(self,
                          train_dataloader,
                          val_dataloader,
                          ):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader


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
        self.ckpt_path = self.config['ckpt_path']


    def load_checkpoint(self):
        if not self.config['resume']:
            self.step = 0
            self.update_step = 0
            self.best_val_loss = float('inf')
            self.optimizer = utils.optimizer_warmup(self.optimizer, self.base_lr, 1, self.max_warmup_steps)
    
        else:
            if self.trainer_mode == 'fsdp2':
                if self.global_rank == 0:
                    checkpoint = torch.load(self.config['ckpt_path'], map_location="cpu")
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
                    self.language_model,
                    model_state_dict,
                    options=options
                )
            
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
    
            else:
                checkpoint = torch.load(self.config['ckpt_path'], map_location=self.device)
            
                self.language_model.module.load_state_dict(checkpoint["model_state_dict"])
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
                self.step = checkpoint["step"]
                self.update_step = checkpoint["update_step"]
                self.best_val_loss = checkpoint["best_val_loss"]
                self.optimizer = utils.optimizer_warmup(self.optimizer, self.base_lr, 1, self.max_warmup_steps)


    def validate(self):
        self.language_model.eval()
    
        total_loss = torch.zeros(1, device=self.device, dtype=torch.float16)
        count = torch.zeros(1, device=self.device, dtype=torch.float16)
    
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

        total_train_loss = torch.zeros(1, device=self.device, dtype=torch.float16)
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
        
                is_sync_step = (self.step % self.grad_accum_steps == 0)
                current_update = self.step // self.grad_accum_steps
        
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    if self.trainer_mode == 'fsdp2':
                        self.language_model.set_requires_gradient_sync(is_sync_step)

                    else:
                        self.language_model.require_backward_grad_sync = is_sync_step
        
                    outputs = self.language_model(*inputs)
                    loss = self.loss_fn(outputs.view(-1, self.vocab_len), labels.view(-1))
        
                    loss_accum_steps += loss.detach()
                    valid_batch_count += 1.0

                    scaled_loss = loss / self.grad_accum_steps
                    self.scaler.scale(scaled_loss).backward()

                if is_sync_step:
                    self.scaler.unscale_(self.optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.language_model.parameters(), max_norm=1.0)

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
                            self.max_steps,
                            self.max_warmup_steps,
                            min_lr=self.min_lr,
                        )
        
                    if self.update_step % self.val_step == 0:
                        val_loss = self.validate()
        
                        dist.all_reduce(total_train_loss, op=dist.ReduceOp.SUM)
                        total_train_loss /= (self.world_size * self.val_step)
        
                        if self.global_rank == 0:
                            print(f'\n Step {self.update_step}: avg train loss {total_train_loss.item()}, avg val loss {val_loss}', flush=True)
        
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                        
                            if self.trainer_mode == 'fsdp2':
                                options = StateDictOptions(full_state_dict=True, cpu_offload=True)
                        
                                model_state_dict = get_model_state_dict(self.language_model, options=options)
                                optimizer_state_dict = get_optimizer_state_dict(self.language_model, self.optimizer, options=options)
                        
                            elif self.global_rank == 0:
                                model_state_dict = self.language_model.module.state_dict()
                                optimizer_state_dict = self.optimizer.state_dict()
                        
                            if self.global_rank == 0:
                                checkpoint = {
                                    'model_state_dict': model_state_dict,
                                    'optimizer_state_dict': optimizer_state_dict,
                                    'scaler_state_dict': self.scaler.state_dict(),
                                    'step': self.step,
                                    'update_step': self.update_step,
                                    'best_val_loss': self.best_val_loss,
                                }
                        
                                torch.save(checkpoint, self.ckpt_path)
            
                        total_train_loss.zero_()
                        dist.barrier(device_ids=[self.local_rank])
                        self.language_model.train()
        
                    if self.update_step >= self.max_steps:
                        break


