# QA Fine-Tuning Phase

This phase extends the pretrained GPT model into a question-answering system using SQuAD-style data. The model is trained autoregressively from prompts of the form:

Context: ...
Question: ...
Answer: ...

Training loss is applied only to the answer portion of each sample, while the context and question are used as conditioning information. A block causal mask is used so each QA example attends only within its own `Context:` block.

## Table of Contents
- [Current Test Performance](#current-test-performance)
- [Training](#training)
  - [Gradient Accumulation](#gradient-accumulation)
- [Testing / Inference](#testing--inference)
- [Training vs. Inference](#training-vs-inference)
- [Configuration](#configuration)
- [Current Status](#current-status)
  
## Current Test Performance

Using the fully fine-tuned model on the test set:

```text
Exact Match (EM): 43.10%
F1 Score:         48.53%
```

The evaluation includes both answerable and unanswerable questions.

EM measures the percentage of predictions that exactly match the normalized reference answer. F1 is token based and gives partial credit when the generated answer overlaps with the reference without matching it exactly.

For example:

```text
Prediction:   Mahasanghika school
Ground truth: Mahasanghika
```

would fail Exact Match but still receive substantial F1 credit.

## Training

Training is distributed across four GPUs using PyTorch Distributed Data Parallel.

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

### Gradient Accumulation

The training code also uses gradient accumulation to obtain a larger effective batch size than can fit into GPU memory at once.

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

Therefore the approximate effective token batch per optimizer update is:

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


Inference
---------
1 GPU
no gradients
no optimizer
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
train_eval_squad.py test  config.json
```

The first argument selects the execution phase, while the second specifies the configuration file.

## Current Status

The QA pipeline currently supports:

- SQuAD-style context/question/answer formatting
- answerable and unanswerable questions
- answer-only loss masking
- blocked causal attention between independent QA samples
- distributed multi-GPU training
- gradient accumulation for large effective batch sizes
- single-GPU autoregressive inference
- Exact Match evaluation
- token-level F1 evaluation
- configuration-driven training and testing

The current full-model QA checkpoint achieves:

```text
EM: 43.10%
F1: 48.53%
```

on the test evaluation.
