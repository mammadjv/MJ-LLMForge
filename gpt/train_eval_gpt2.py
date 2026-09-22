import os
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.distributed.fsdp import fully_shard
from torch.nn.parallel import DistributedDataParallel as DDP
import bitsandbytes as bnb

import tiktoken
from collections import OrderedDict

import gpt_dataset
from gpt import GPT
from transformer import utils, trainer


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


def run_train(config, device, local_rank, global_rank, world_size):
    base_lr = config["base_lr"]
    seq_len = config["seq_len"]
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]

    iter_batch_size = config["iter_batch_size"]

    use_checkpointing = config["use_checkpointing"]
    use_flash = config["use_flash"]

    data_dir = config["data_dir"]
    num_workers = config['num_workers']
    trainer_mode = config['trainer_mode']

    gpt_train_dataset = gpt_dataset.GPTDataset(data_dir, 'train', iter_batch_size, seq_len, global_rank, world_size, num_workers)
    gpt_val_dataset = gpt_dataset.GPTDataset(data_dir, 'val', iter_batch_size, seq_len, global_rank, world_size, num_workers)

    vocab_len = gpt_train_dataset.vocab_len
    config["vocab_len"] = vocab_len

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

    loss_fn = nn.CrossEntropyLoss()
    llm_trainer = trainer.Trainer(language_model=language_model,
                      loss_fn=loss_fn,
                      device=device,
                      global_rank=global_rank,
                      world_size=world_size,
                      local_rank=local_rank,
                      config=config,
                      trainer_mode=trainer_mode,
                     )

    gpt_train_dataset.update_dataset_sharding(llm_trainer.update_step,
                                              llm_trainer.grad_accum_steps,
                                             )

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
    llm_trainer.update_dataloaders(train_dataloader,
                            val_dataloader,
                           )
    
    llm_trainer.start()


def run_test(config, device, local_rank, global_rank, world_size):
    seq_len = config["seq_len"]
    embed_dim = config["embed_dim"]
    num_heads = config["num_heads"]
    num_layers = config["num_layers"]
    use_checkpointing = config["use_checkpointing"]
    use_flash = config["use_flash"]
    ckpt_path = config["best_ckpt_path"]

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