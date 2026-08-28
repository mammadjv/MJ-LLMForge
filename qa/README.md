# QA Fine-Tuning Phase

This phase extends the pretrained GPT model into a question-answering system using SQuAD-style data. The model is trained autoregressively from prompts of the form:

```text
Context: ...
Question: ...
Answer: ...
```

Training loss is applied only to the answer portion of each sample, while the context and question are used as conditioning information. A block causal mask is used so each QA example attends only within its own `Context:` block.

The QA pipeline supports both traditional fine-tuning and parameter-efficient fine-tuning using Low-Rank Adaptation (LoRA).

## Table of Contents

- [Fine-Tuning Methods](#fine-tuning-methods)
  - [Traditional Fine-Tuning](#traditional-fine-tuning)
  - [LoRA Fine-Tuning](#lora-fine-tuning)
  - [Applying LoRA for Inference](#applying-lora-for-inference)
- [Test Performance](#test-performance)
- [Training](#training)
  - [Gradient Accumulation](#gradient-accumulation)
- [Testing / Inference](#testing--inference)
- [Training vs. Inference](#training-vs-inference)
- [Configuration](#configuration)
- [Current Status](#current-status)

## Fine-Tuning Methods

The training pipeline supports two fine-tuning strategies controlled through the configuration file:

- LoRA fine-tuning using `lora_rank`
- Traditional fine-tuning using `n_frozen_layers`

The selected strategy depends on the value of `lora_rank`.

### Traditional Fine-Tuning

When:

```text
lora_rank = 0
```

LoRA is disabled.

The value of:

```text
n_frozen_layers
```

determines how many of the initial Transformer layers are frozen during fine-tuning.

The remaining layers are trained normally using backpropagation.

This allows experiments with different amounts of the pretrained model kept fixed while adapting the remaining parameters to the QA task.

### LoRA Fine-Tuning

When:

```text
lora_rank > 0
```

the pretrained model parameters are frozen and Low-Rank Adaptation (LoRA) is enabled.

Instead of directly updating the pretrained attention projection matrices, LoRA learns a low-rank update:

```text
ΔW = B A
```

where:

```text
A ∈ R^(r × d)
B ∈ R^(d × r)
```

and `r` is specified by `lora_rank`.

For an original projection:

```text
y = Wx
```

the LoRA-adapted projection becomes:

```text
y = Wx + scale · BAx
```

This allows the model to adapt to the QA task while training only a small subset of additional parameters.

LoRA adapters are applied to the query, key, and value projections of the attention layers.

Experiments were performed using:

```text
lora_rank = 8
lora_rank = 16
lora_rank = 32
```

### Applying LoRA for Inference

The model implements an `apply_lora()` function that merges the learned low-rank updates into the original attention projection weights.

Conceptually, the merge performs:

```text
Wq ← Wq + scale · (Bq @ Aq)
Wk ← Wk + scale · (Bk @ Ak)
Wv ← Wv + scale · (Bv @ Av)
```

After merging, inference can use the modified projection matrices directly rather than computing the LoRA branch separately during every forward pass.

The inference pipeline checks `lora_rank`.

If:

```text
lora_rank > 0
```

the corresponding LoRA weights are loaded and `apply_lora()` is used to merge the adapters into the pretrained model.

If:

```text
lora_rank = 0
```

LoRA is not applied. The pipeline instead uses `n_frozen_layers` for the traditional fine-tuning configuration.

## Test Performance

Both LoRA and traditional fine-tuning were evaluated using Exact Match (EM) and token-level F1.

### LoRA Results

| LoRA Rank | Exact Match (EM) | F1 |
|----------:|-----------------:|---:|
| 8 | 35.52% | 39.94% |
| 16 | 43.44% | 47.69% |
| **32** | **47.40%** | **52.91%** |

Increasing the LoRA rank consistently improved QA performance in these experiments.

Rank 32 produced the strongest LoRA result:

```text
EM: 47.40%
F1: 52.91%
```

### Traditional Fine-Tuning Results

Two traditional fine-tuning configurations were evaluated:

| Configuration | Exact Match (EM) | F1 |
|--------------|-----------------:|---:|
| `best_model_qa_all_lora_8.pt` | **47.12%** | **52.54%** |
| `best_model_qa_all_lora_16.pt` | 34.56% | 38.36% |

Despite the checkpoint names containing `lora`, these two models were trained using traditional fine-tuning rather than LoRA.

### LoRA vs. Traditional Fine-Tuning

The strongest result from each approach was:

| Method | Configuration | Exact Match (EM) | F1 |
|---|---|---:|---:|
| Traditional Fine-Tuning | `best_model_qa_all_lora_8.pt` | 47.12% | 52.54% |
| **LoRA** | **Rank 32** | **47.40%** | **52.91%** |

The rank-32 LoRA model achieved performance comparable to the strongest traditionally fine-tuned model:

```text
EM difference: +0.28 percentage points
F1 difference: +0.37 percentage points
```

The difference is small enough that the two approaches should be considered approximately equivalent in test performance based on these experiments.

However, LoRA reaches this performance while updating only the low-rank adapter parameters rather than directly fine-tuning the pretrained model parameters.

The results also show a clear effect of LoRA rank:

```text
Rank 8
EM 35.52 / F1 39.94
        ↓
Rank 16
EM 43.44 / F1 47.69
        ↓
Rank 32
EM 47.40 / F1 52.91
```

For this QA task, increasing the adapter rank provided substantially more adaptation capacity, with rank 32 giving the best result among the evaluated LoRA configurations.

### Evaluation Metrics

The evaluation includes both answerable and unanswerable questions.

Exact Match measures whether the normalized generated answer exactly matches the normalized reference answer.

F1 is token based and provides partial credit when the generated answer overlaps with the reference answer without matching it exactly.

For example:

```text
Prediction:   Mahasanghika school
Ground truth: Mahasanghika
```

would fail Exact Match but can still receive substantial F1 credit.

## Training

Training is distributed across four GPUs using PyTorch Distributed Data Parallel (DDP).

Run:

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$SLURM_NODEID \
    train_eval_squad.py train config.json
```

`--nproc_per_node=4` launches four processes, with one process assigned to each GPU.

Each GPU processes its own local mini-batch. DDP synchronizes gradients across the four replicas during optimizer updates so that the replicas remain identical.

When LoRA is enabled, only the LoRA adapter parameters require gradients. The pretrained model parameters remain frozen.

When LoRA is disabled, `n_frozen_layers` controls how many initial Transformer layers remain frozen during traditional fine-tuning.

### Gradient Accumulation

The training code uses gradient accumulation to obtain a larger effective batch size than can fit into GPU memory at once.

Instead of performing an optimizer update after every mini-batch, gradients are accumulated across several iterations:

```text
micro-batch 1
    ↓
backward
    ↓
accumulate gradients

micro-batch 2
    ↓
backward
    ↓
accumulate gradients

...

final accumulation step
    ↓
synchronize gradients across GPUs
    ↓
optimizer step
    ↓
zero gradients
```

The accumulation count is determined from the requested global token batch size:

```python
grad_accum_steps = int(
    actual_gpt_batch_size /
    (seq_len * iter_batch_size * world_size)
)
```

Therefore, the approximate effective token batch per optimizer update is:

```text
sequence length
× local batch size
× number of GPUs
× gradient accumulation steps
```

For example, with:

```text
seq_len           = 1024
iter_batch_size   = 1
world_size        = 4
grad_accum_steps  = 8
```

the optimizer sees:

```text
1024 × 1 × 4 × 8
= 32,768 tokens/update
```

or 32 packed sequences of length 1024 per optimizer update.

Gradient synchronization is only required on the final accumulation step, reducing unnecessary DDP communication during intermediate accumulation steps.

## Testing / Inference

Inference only requires a single GPU because there is no gradient computation and no need to synchronize model replicas.

Run:

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank=$SLURM_NODEID \
    train_eval_squad.py test config.json
```

The test loader produces one QA prompt at a time containing:

```text
Context: ...
Question: ...
Answer:
```

The model then generates tokens autoregressively until the `<|endoftext|>` token is produced or the generation limit is reached.

For LoRA checkpoints, the pretrained base model and learned LoRA parameters are loaded and the low-rank updates are merged using `apply_lora()` before evaluation.

Using one GPU for inference avoids unnecessarily loading identical copies of the model onto four GPUs. Multi-GPU training is useful because gradient computation is expensive and can be distributed, while single-example autoregressive inference does not benefit from the same DDP strategy.

## Training vs. Inference

```text
Training
--------
4 GPUs
DistributedDataParallel
gradient computation
gradient accumulation
cross-GPU gradient synchronization
optimizer updates

LoRA:
    frozen pretrained model
    trainable low-rank adapters

Traditional:
    n_frozen_layers frozen
    remaining model parameters trainable


Inference
---------
1 GPU
no gradients
no optimizer
LoRA adapters merged when enabled
autoregressive token generation
EM/F1 evaluation
```

## Configuration

Training and testing parameters are read from:

```text
config.json
```

The same script is used for both phases:

```text
train_eval_squad.py train config.json
train_eval_squad.py test config.json
```

The first argument selects the execution phase, while the second specifies the configuration file.

The fine-tuning strategy is controlled primarily by:

```text
lora_rank
n_frozen_layers
```

When:

```text
lora_rank > 0
```

LoRA fine-tuning is enabled.

When:

```text
lora_rank = 0
```

traditional fine-tuning is used and `n_frozen_layers` determines the number of frozen Transformer layers.

## Current Status

The QA pipeline currently supports:

- SQuAD-style context/question/answer formatting
- answerable and unanswerable questions
- answer-only loss masking
- blocked causal attention between independent QA samples
- traditional partial/full-model fine-tuning
- LoRA fine-tuning of attention projections
- configurable LoRA rank
- configurable number of frozen layers for traditional fine-tuning
- LoRA weight merging through `apply_lora()`
- distributed multi-GPU training
- gradient accumulation for large effective batch sizes
- single-GPU autoregressive inference
- Exact Match evaluation
- token-level F1 evaluation
- configuration-driven training and testing

The strongest LoRA configuration evaluated so far is rank 32:

```text
LoRA rank: 32
EM:        47.40%
F1:        52.91%
```

which achieved approximately the same QA performance as the strongest traditionally fine-tuned checkpoint:

```text
Traditional fine-tuning
EM: 47.12%
F1: 52.54%
```
