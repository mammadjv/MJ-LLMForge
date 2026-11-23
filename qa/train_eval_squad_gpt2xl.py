#!/usr/bin/env python3
"""GPT-2 XL SQuAD baseline: attention LoRA or partial fine-tuning.

Training and validation use the ORIGINAL qa_dataset.QADataset packed batches,
block-diagonal 4D causal attention mask, and answer-only next-token labels.
Testing also uses the original QADataset test split.

Requires transformers>=4.55,<5, peft, torch, tiktoken, numpy.
Training: torchrun --standalone --nproc_per_node=4 train_eval_squad_gpt2xl_packed.py \
             --phase train --config config.json --lora_rank 16 --n_frozen_layers 0
Testing:  torchrun --standalone --nproc_per_node=1 train_eval_squad_gpt2xl_packed.py \
             --phase test --config config.json --lora_rank 16
"""

import argparse
import json
import math
import os
import re
import string
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F
import tiktoken

import qa_dataset
from transformers import GPT2LMHeadModel
from peft import LoraConfig, TaskType, get_peft_model
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict


def setup_distributed():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required (V100 supports fp16, not bf16).")
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # Always launch with torchrun, including one-process evaluation.
        dist.init_process_group("nccl", init_method="env://")
    return rank, local_rank, world_size, torch.device("cuda", local_rank)


def log(msg, rank=0):
    if rank == 0:
        print(msg, flush=True)


def packed_batch_on_device(batch, device):
    """Unpack QADataset's ((x, block_mask), y) without altering its semantics.

    x:    [B,T] input tokens
    mask: [B,1,T,T] additive 0/-inf mask, including block boundaries
    y:    [B,T] already-shifted next-token labels; -1 = ignore
    """
    (x, block_mask), y = batch
    x = x.to(device, non_blocking=True)
    block_mask = block_mask.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    if x.ndim != 2 or y.shape != x.shape or block_mask.shape != (x.shape[0], 1, x.shape[1], x.shape[1]):
        raise ValueError(f"Packed dataset shape mismatch: x={tuple(x.shape)}, mask={tuple(block_mask.shape)}, y={tuple(y.shape)}")
    return x, block_mask, y


def packed_answer_loss(model, batch, device):
    """Match the custom trainer: logits at x[t] predict y[t].

    Hugging Face model(labels=...) shifts internally. Do NOT pass these labels
    to the model or they would be shifted twice.
    """
    x, block_mask, y = packed_batch_on_device(batch, device)
    if torch.any(x >= 50257):
        raise ValueError("Packed data contains token IDs outside GPT-2 XL's 50,257-token vocabulary")
    with torch.autocast("cuda", dtype=torch.float16):
        logits = model(input_ids=x, attention_mask=block_mask, use_cache=False).logits
        # Sum over answer tokens, including EOT, exactly as the original QADataset specifies.
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-1,
            reduction="sum",
        )
    target_count = int((y != -1).sum().item())
    if target_count == 0:
        raise RuntimeError("QADataset yielded a batch with no answer tokens")
    return loss_sum / target_count, target_count


def normalize_answer(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def compute_em(pred, truth):
    return int(normalize_answer(pred) == normalize_answer(truth))


def compute_f1(pred, truth):
    a, b = normalize_answer(pred).split(), normalize_answer(truth).split()
    if not a or not b:
        return float(a == b)
    common = sum((Counter(a) & Counter(b)).values())
    return 2 * common / (len(a) + len(b)) if common else 0.0


def build_model(config, lora_rank, n_frozen_layers, *, training):
    model_name = config.get("hf_model_name", "openai-community/gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained(
        model_name, torch_dtype=torch.float32, attn_implementation="sdpa"
    )
    model.config.use_cache = not training
    if model.config.n_positions != config["seq_len"]:
        if config["seq_len"] > model.config.n_positions:
            raise ValueError("seq_len exceeds GPT-2 XL's 1024 learned position embeddings")
    if lora_rank > 0:
        # GPT-2 uses a *fused* QKV Conv1D projection called c_attn.
        # Unlike custom separate Q/K/V LoRA, this shares one low-rank basis.
        adapter = LoraConfig(
            r=lora_rank,
            lora_alpha=int(config.get("lora_alpha", lora_rank)),
            lora_dropout=float(config.get("lora_dropout", 0.0)),
            bias="none",
            target_modules=["c_attn"],
            fan_in_fan_out=True,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, adapter)
    else:
        total_blocks = len(model.transformer.h)
        if not (0 <= n_frozen_layers <= total_blocks):
            raise ValueError(f"n_frozen_layers must be 0..{total_blocks}")
        # Same semantics as your original trainer: freeze the FIRST N blocks.
        # Embeddings and final layer norm remain trainable (as in your code).
        for block in model.transformer.h[:n_frozen_layers]:
            for param in block.parameters():
                param.requires_grad_(False)
    if training and bool(config.get("use_checkpointing", True)):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


def transformer_blocks(model, lora_rank):
    base = model.base_model.model if lora_rank > 0 else model
    return base.transformer.h


def shard_model(model, lora_rank, device):
    # Shard each GPT-2 block, then the root model, for peak-memory savings.
    model.to(device)
    mp = MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float32)
    for block in transformer_blocks(model, lora_rank):
        fully_shard(block, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    return model


def lr_for_step(step, base_lr, min_lr, warmup, total):
    if warmup > 0 and step <= warmup:
        return base_lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def evaluate_loss(model, loader, max_batches, device):
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
            mean_loss, valid = packed_answer_loss(model, batch, device)
            totals[0] += mean_loss.detach().float() * valid
            totals[1] += valid
    # Every rank reads identical validation shards and participates in the same collectives.
    dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    model.train()
    return (totals[0] / totals[1].clamp_min(1)).item()


def save_best(model, output_path, metadata, is_lora, rank):
    # Collective; rank 0 receives the standard full CPU state dict.
    options = StateDictOptions(full_state_dict=True, cpu_offload=True)
    full_state = get_model_state_dict(model, options=options)
    if rank == 0:
        if is_lora:
            state = {k: v for k, v in full_state.items() if "lora_A." in k or "lora_B." in k}
            if not state:
                raise RuntimeError("LoRA adapter keys were not found in the full state dict")
        else:
            state = full_state
        checkpoint = dict(metadata)
        checkpoint["model_state_dict" if not is_lora else "adapter_state_dict"] = state
        tmp = output_path + ".tmp"
        torch.save(checkpoint, tmp)
        os.replace(tmp, output_path)
        print(f"Saved best checkpoint: {output_path}", flush=True)
    del full_state
    dist.barrier()


def run_train(config, rank, world_size, device, args):
    lora_rank = int(args.lora_rank if args.lora_rank is not None else config.get("lora_rank", 0))
    frozen = int(args.n_frozen_layers if args.n_frozen_layers is not None else config.get("n_frozen_layers", 0))
    microbatch = int(args.iter_batch_size if args.iter_batch_size is not None else config.get("iter_batch_size", 1))
    seq_len = int(config["seq_len"])
    max_steps = int(args.max_steps if args.max_steps is not None else config.get("max_steps", 10000))
    val_step = int(config.get("val_step", 200))
    accumulation = int(config.get("grad_accum_steps", 0))
    if accumulation <= 0:
        effective_nominal = int(config["actual_gpt_batch_size"])
        accumulation = max(1, effective_nominal // (seq_len * microbatch * world_size))
    output_dir = Path(args.output_dir or config.get("hf_output_dir", "./gpt2xl_qa"))
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"lora_rank{lora_rank}" if lora_rank else f"ft_freeze{frozen}"
    output_path = str(output_dir / f"{name}_best.pt")

    model = build_model(config, lora_rank, frozen, training=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    log(f"GPT-2 XL baseline | {name} | trainable={trainable:,}/{all_params:,} ({trainable / all_params:.3%})", rank)
    log(f"Packed QADataset | accumulation={accumulation} | packed sequences/GPU={microbatch} | nominal tokens/update={accumulation * microbatch * world_size * seq_len}", rank)
    model = shard_model(model, lora_rank, device)

    decay, no_decay = [], []
    for pname, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and "lora_" not in pname:
            decay.append(p)
        else:
            no_decay.append(p)
    optim = torch.optim.AdamW(
        [{"params": decay, "weight_decay": 0.1},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=float(config.get("base_lr", 5e-5)),
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda")

    # Use exactly the same dataset class, packed sequence formation, and attention mask
    # as train_eval_squad.py. QADataset already applies iter_batch_size internally.
    train_ds = qa_dataset.QADataset(config["data_dir"], "train", microbatch, seq_len, rank, world_size)
    val_ds = qa_dataset.QADataset(config["data_dir"], "val", microbatch, seq_len, rank, world_size)
    workers = int(config.get("num_workers", 0))
    loader_kwargs = dict(batch_size=None, num_workers=workers, pin_memory=True)
    if workers:
        loader_kwargs.update(prefetch_factor=2, persistent_workers=True)
    train_loader = DataLoader(train_ds, **loader_kwargs)
    # Every rank uses the same validation shard; avoid multi-worker duplication.
    val_loader = DataLoader(val_ds, batch_size=None, num_workers=0)
    train_iter = iter(train_loader)

    base_lr = float(config.get("base_lr", 5e-5))
    min_lr = float(config.get("min_lr", base_lr * 0.1))
    warmup = int(config.get("max_warmup_steps", 100))
    max_val_batches = int(config.get("hf_val_batches", 50))
    grad_clip = float(config.get("grad_clip", 1.0))
    best_val = float("inf")
    log("Starting training", rank)
    model.train()
    for step in range(1, max_steps + 1):
        lr = lr_for_step(step, base_lr, min_lr, warmup, max_steps)
        for group in optim.param_groups:
            group["lr"] = lr
        optim.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accumulation):
            batch = next(train_iter)
            loss, _ = packed_answer_loss(model, batch, device)
            step_loss += loss.detach().item() / accumulation
            scaler.scale(loss / accumulation).backward()
        # Synchronization remains enabled on every micro-batch for FSDP2
        # memory safety (slower than suppressing intermediate reductions).
        scaler.unscale_(optim)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        old_scale = scaler.get_scale()
        scaler.step(optim)
        scaler.update()
        if step == 1 or step % int(config.get("hf_log_every", 20)) == 0:
            log(f"Step {step}/{max_steps}: train_loss={step_loss:.4f} grad_norm={float(norm):.3f} lr={lr:.3g} scale={old_scale:.0f}->{scaler.get_scale():.0f}", rank)
        if step % val_step == 0 or step == max_steps:
            val_loss = evaluate_loss(model, val_loader, max_val_batches, device)
            log(f"Step {step}: val_loss={val_loss:.5f}, best_val={best_val:.5f}", rank)
            if val_loss < best_val:
                best_val = val_loss
                metadata = {
                    "hf_model_name": config.get("hf_model_name", "openai-community/gpt2-xl"),
                    "step": step,
                    "val_loss": val_loss,
                    "lora_rank": lora_rank,
                    "lora_alpha": int(config.get("lora_alpha", lora_rank)),
                    "lora_dropout": float(config.get("lora_dropout", 0.0)),
                    "n_frozen_layers": frozen,
                    "trainable_parameters": trainable,
                    "model_type": "gpt2-xl",
                }
                save_best(model, output_path, metadata, lora_rank > 0, rank)
    log(f"Complete. Best validation loss={best_val:.5f} checkpoint={output_path}", rank)


def load_for_inference(config, checkpoint_path, device, rank):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lora_rank = int(ckpt["lora_rank"])
    frozen = int(ckpt.get("n_frozen_layers", 0))
    ck_config = dict(config)
    ck_config["hf_model_name"] = ckpt["hf_model_name"]
    ck_config["lora_alpha"] = ckpt.get("lora_alpha", lora_rank)
    ck_config["lora_dropout"] = ckpt.get("lora_dropout", 0.0)
    model = build_model(ck_config, lora_rank, frozen, training=False)
    if lora_rank:
        adapters = ckpt["adapter_state_dict"]
        all_keys = set(dict(model.named_parameters()))
        missing_adapters = set(adapters) - all_keys
        if missing_adapters:
            raise RuntimeError(f"Unrecognized adapter keys: {list(missing_adapters)[:5]}")
        all_adapters = {k for k in all_keys if "lora_A." in k or "lora_B." in k}
        if set(adapters) != all_adapters:
            raise RuntimeError(f"Adapter key mismatch: expected {len(all_adapters)}, got {len(adapters)}")
        model.load_state_dict(adapters, strict=False)
        model = model.merge_and_unload()
    else:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device).eval()
    log(f"Loaded {checkpoint_path} (LoRA rank {lora_rank}; frozen first {frozen})", rank)
    return model, ckpt


def run_test(config, device, args, rank):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("Use --nproc_per_node=1 for autoregressive evaluation")
    lora_rank = int(args.lora_rank if args.lora_rank is not None else config.get("lora_rank", 0))
    frozen = int(args.n_frozen_layers if args.n_frozen_layers is not None else config.get("n_frozen_layers", 0))
    name = f"lora_rank{lora_rank}" if lora_rank else f"ft_freeze{frozen}"
    directory = args.output_dir or config.get("hf_output_dir", "./gpt2xl_qa")
    checkpoint_path = args.checkpoint or str(Path(directory) / f"{name}_best.pt")
    model, ckpt = load_for_inference(config, checkpoint_path, device, rank)
    encoding = tiktoken.get_encoding("gpt2")
    eot = encoding.eot_token
    seq_len = min(config["seq_len"], 1024)
    test_limit = args.test_limit if args.test_limit is not None else config.get("test_limit")
    max_new_tokens = int(config.get("test_max_new_tokens", 150))
    test_dataset = qa_dataset.QADataset(config["data_dir"], "test", 1, seq_len, 0, 1)
    test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0)
    em_total = f1_total = count = 0
    ans_count = no_count = 0
    ans_em = ans_f1 = no_em = no_f1 = 0.0
    trunc = 0
    with torch.inference_mode():
        for idx, (prompt, ground_truth) in enumerate(test_loader):
            if test_limit is not None and idx >= int(test_limit):
                break
            x = prompt.to(device=device, dtype=torch.long)
            initial_length = x.shape[1]
            if initial_length > seq_len:
                trunc += 1
            if x.max().item() >= model.config.vocab_size:
                raise ValueError("Prompt has an out-of-vocabulary GPT-2 XL token ID")
            for _ in range(max_new_tokens):
                context = x[:, -seq_len:]
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(input_ids=context, use_cache=False).logits
                next_id = logits[:, -1].argmax(-1, keepdim=True)
                x = torch.cat((x, next_id), dim=1)
                if next_id.item() == eot:
                    break
            predicted = x[0, initial_length:].tolist()
            reference = ground_truth[0].tolist()
            if eot in predicted:
                predicted = predicted[:predicted.index(eot)]
            if eot in reference:
                reference = reference[:reference.index(eot)]
            prediction = encoding.decode(predicted)
            truth = encoding.decode(reference)
            em = compute_em(prediction, truth)
            f1 = compute_f1(prediction, truth)
            count += 1
            em_total += em
            f1_total += f1
            if normalize_answer(truth):
                ans_count += 1
                ans_em += em
                ans_f1 += f1
            else:
                no_count += 1
                no_em += em
                no_f1 += f1
            if count == 1 or count % 1000 == 0:
                log(f"{name}: {count} examples | answerable={ans_count} unanswerable={no_count} EM={100*em_total/count:.2f} F1={100*f1_total/count:.2f}", rank)
    result = {
        "checkpoint": checkpoint_path,
        "model": "openai-community/gpt2-xl",
        "lora_rank": ckpt["lora_rank"],
        "n_frozen_layers": ckpt.get("n_frozen_layers", 0),
        "examples": count,
        "em": 100*em_total/max(1,count), "f1": 100*f1_total/max(1,count),
        "answerable_examples": ans_count,
        "answerable_em": 100*ans_em/max(1,ans_count),
        "answerable_f1": 100*ans_f1/max(1,ans_count),
        "unanswerable_examples": no_count,
        "unanswerable_em": 100*no_em/max(1,no_count),
        "unanswerable_f1": 100*no_f1/max(1,no_count),
        "prompts_longer_than_context": trunc,
    }
    log(json.dumps(result, indent=2), rank)
    output_dir = Path(args.output_dir or config.get("hf_output_dir", "./gpt2xl_qa"))
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / f"{name}_qa_inference_results.json", "w") as f:
        json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="GPT-2 XL SQuAD LoRA and partial fine-tuning baseline")
    parser.add_argument("--phase", choices=["train", "test"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--n_frozen_layers", type=int, default=None)
    parser.add_argument("--iter_batch_size", type=int, default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--test_limit", type=int, default=None)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    if int(config["seq_len"]) > 1024:
        raise ValueError("GPT-2 XL has 1024 positions, but config seq_len > 1024")
    rank, local_rank, world_size, device = setup_distributed()
    # Identical LoRA initialization on every rank is essential before FSDP2 sharding.
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        if args.phase == "train":
            run_train(config, rank, world_size, device, args)
        else:
            run_test(config, device, args, rank)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
