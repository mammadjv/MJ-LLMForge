import os
import argparse
import json
import sys
import time
import re
import string
from collections import Counter

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
import tiktoken

import qa_dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from gpt.gpt import GPT
from transformer import utils
from transformer import trainer


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
    trainer_mode = config['trainer_mode']
    n_frozen_layers = config['n_frozen_layers']

    iter_batch_size = config['iter_batch_size']
    actual_gpt_batch_size = config['actual_gpt_batch_size']

    use_checkpointing = config['use_checkpointing']
    use_flash = config['use_flash']

    max_steps = config['max_steps']
    max_warmup_steps = config['max_warmup_steps']
    val_step = config['val_step']
    num_workers = config['num_workers']
    data_dir = config['data_dir']
    lora_rank = config['lora_rank']

    qa_train_dataset = qa_dataset.QADataset(data_dir, 'train', iter_batch_size, seq_len, local_rank, world_size)
    qa_val_dataset = qa_dataset.QADataset(data_dir, 'val', iter_batch_size, seq_len, local_rank, world_size)
    vocab_len = qa_dataset.get_vocab_len()
    config["vocab_len"] = vocab_len

    language_model = GPT(
        dictionary_length=vocab_len,
        seq_len=seq_len,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        use_checkpointing=use_checkpointing,
        use_flash=use_flash,
        lora_rank=lora_rank,
    )
    language_model = language_model.to(device)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1)

    llm_trainer = trainer.Trainer(language_model=language_model,
                      loss_fn=loss_fn,
                      device=device,
                      global_rank=global_rank,
                      world_size=world_size,
                      local_rank=local_rank,
                      config=config,
                      trainer_mode=trainer_mode,
                     )

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

    llm_trainer.update_dataloaders(train_dataloader,
                                   val_dataloader,
                                  )
    llm_trainer.start()


def run_test(config, device, local_rank):
    seq_len = config['seq_len']
    embed_dim = config['embed_dim']
    num_heads = config['num_heads']
    num_layers = config['num_layers']
    use_checkpointing = config['use_checkpointing']
    use_flash = config['use_flash']
    data_dir = config['data_dir']

    checkpoints = [
        ('./lora_weights_32.pt', 32),
        ('./lora_weights_8.pt', 8),
        ('./lora_weights_16.pt', 16),
        ('./best_model_nxl5_frozen_8.pt', 0),
        ('./best_model_nxl5_frozen_16.pt', 0),
    ]

    for ckpt_path, lora_rank in checkpoints:
        print("\n" + "=" * 60)
        print("Testing checkpoint:", ckpt_path)
        print("=" * 60)

        language_model = GPT(
            dictionary_length=vocab_len,
            seq_len=seq_len,
            num_layers=num_layers,
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_checkpointing=use_checkpointing,
            use_flash=use_flash,
            lora_rank=lora_rank,
        )

        # Traditional Finetuning
        if lora_rank == 0:
            language_model.load_state_dict(
                load_state_dict_strip_module_qa(ckpt_path, map_location="cpu"),
                strict=False,
            )

        else:
            language_model.load_state_dict(
                load_state_dict_strip_module_qa(gpt_path,
                                                 map_location="cpu",
                                                ),
                strict=False,
            )

            # Load adapter
            lora_checkpoint = torch.load(ckpt_path, map_location="cpu")
            language_model.load_state_dict(
                lora_checkpoint["model_state_dict"],
                strict=False,
            )
            language_model.apply_lora()

        language_model = language_model.to(device)
        language_model = DDP(
            language_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

        language_model.eval()

        test_dataset = qa_dataset.QADataset(
            data_dir=data_dir,
            split='test',
            batch_size=1,
            seq_len=seq_len,
            rank=0,
            world_size=1
        )

        test_loader = DataLoader(test_dataset, batch_size=1)
        eot_token = test_dataset.word_encoding.eot_token

        with torch.no_grad():
            total_em = 0
            total_f1 = 0
            num_examples = 0

            for idx, (prompt, ground_truth) in enumerate(test_loader):
                x = prompt.to(device=device, dtype=torch.long)
                prompt_len = x.size(1)

                for j in range(150):
                    curr_T = x.size(1)
                    causal = torch.tril(torch.ones(curr_T, curr_T, device=device, dtype=torch.bool))
                    attn_mask = torch.where(causal, 0.0, float("-inf")).unsqueeze(0).unsqueeze(1)
                    begin = time.time()
                    logits = language_model(x, attn_mask)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    x = torch.cat([x, next_token], dim=1)
                    if next_token.item() == eot_token:
                        break

                generated_ids = x[0, prompt_len:].cpu().tolist()
                generated_text = test_dataset.word_encoding.decode(generated_ids)

                ground_truth_ids = ground_truth[0].cpu().tolist()
                ground_truth_text = test_dataset.word_encoding.decode(ground_truth_ids)

                generated_clean = generated_text.replace("<|endoftext|>", "")
                ground_truth_clean = ground_truth_text.replace("<|endoftext|>", "")

                em = compute_em(generated_clean, ground_truth_clean)
                f1 = compute_f1(generated_clean, ground_truth_clean)

                total_em += em
                total_f1 += f1
                num_examples += 1

                if idx % 3000 == 0:
                    print(test_dataset.word_encoding.decode(x[0].cpu().tolist()))
                    print("Generated answer:{}".format(generated_text), flush=True)
                    print("Ground truth:{}".format(ground_truth_text), flush=True)
                    print("EM: {}, F1: {:.4f}".format(em, f1), flush=True)
                    print("-" * 30, flush=True)

            avg_em = 100 * total_em / num_examples
            avg_f1 = 100 * total_f1 / num_examples

            print("\nCheckpoint:", ckpt_path, flush=True)
            print("EM: {:.2f}%".format(avg_em), flush=True)
            print("F1: {:.2f}%".format(avg_f1), flush=True)

        del language_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--phase", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--n_frozen_layers", type=int, default=None)
    parser.add_argument("--iter_batch_size", type=int, default=None)

    args = parser.parse_args()

    setup()

    print('file = ', args.config)
    with open(args.config, "r") as f:
        config = json.load(f)

    # Override config only if explicitly provided
    if args.lora_rank is not None:
        config["lora_rank"] = args.lora_rank

    if args.n_frozen_layers is not None:
        config["n_frozen_layers"] = args.n_frozen_layers

    if args.iter_batch_size is not None:
        config["iter_batch_size"] = args.iter_batch_size

    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = dist.get_world_size()
    global_rank = dist.get_rank()
    device = torch.device(f"cuda:{local_rank}")
    vocab_len = qa_dataset.get_vocab_len()

    if args.phase == "train":
        run_train(config, device, local_rank, global_rank, world_size)

    else:
        run_test(config, device, local_rank)

    cleanup()
