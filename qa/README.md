# SQuAD 2.0 Question Answering — Fine-Tuning a GPT Trained from Scratch

**A controlled comparison of Low-Rank Adaptation (LoRA) and partial Transformer fine-tuning for context-conditioned question answering.**

This phase of [MJ-LLMForge](https://github.com/mammadjv/MJ-LLMForge) adapts a custom, pretrained, decoder-only GPT to SQuAD-style QA. The implementation includes answer-only supervised learning, block-causal masking for packed examples, multi-GPU training, low-rank attention adapters, adapter merging, and autoregressive evaluation of both answerable and unanswerable questions.

> **Scope of reported results:** These are **project-internal results on 3,000 questions from a custom, shard-based holdout of the SQuAD 2.0 training split**. They are **not official SQuAD 2.0 development or test-set scores**. All five checkpoints were evaluated on the same subset with the same decoding and scoring procedure.

## Table of Contents

- [Overview and key findings](#overview-and-key-findings)
- [Evaluation results](#evaluation-results)
  - [Aggregate Exact Match and F1](#aggregate-exact-match-and-f1)
  - [Answerable vs. unanswerable questions](#answerable-vs-unanswerable-questions)
  - [What the score differences mean](#what-the-score-differences-mean)
- [Model and supervised QA objective](#model-and-supervised-qa-objective)
  - [Prompt representation](#prompt-representation)
  - [Answer-only loss and block-causal masking](#answer-only-loss-and-block-causal-masking)
- [Fine-tuning methods](#fine-tuning-methods)
  - [Partial conventional fine-tuning](#partial-conventional-fine-tuning)
  - [Low-Rank Adaptation (LoRA)](#low-rank-adaptation-lora)
  - [Checkpoints](#checkpoints)
- [Dataset and evaluation protocol](#dataset-and-evaluation-protocol)
  - [Tokenization, sharding, and splits](#tokenization-sharding-and-splits)
  - [Autoregressive evaluation and metrics](#autoregressive-evaluation-and-metrics)
- [Distributed training and loss analysis](#distributed-training-and-loss-analysis)
  - [Training configuration](#training-configuration)
  - [Training-loss curves](#training-loss-curves)
  - [Validation-loss curves](#validation-loss-curves)
  - [Checkpoint selection and overfitting](#checkpoint-selection-and-overfitting)
- [Reproduction](#reproduction)

## Overview and key findings

The 3,000-example evaluation contains **2,461 answerable questions (82.03%)** and **539 unanswerable questions (17.97%)**. A correct answer to an unanswerable question is an **empty generated response** (after normalization), represented in the training data by an empty answer field followed by the end-of-text token.

Three observations define the comparison:

1. **LoRA rank 16 has the highest aggregate scores:** **60.20% Exact Match (EM)** and **67.16% token-level F1**. LoRA rank 32 is extremely close at 60.03% EM / 66.86% F1.
2. **Partial fine-tuning with the first 16 layers frozen is strongest on answerable questions:** **62.33% answerable EM** and **71.04% answerable F1**. However, it correctly abstains on only **36.73%** of unanswerable examples.
3. **Abstention changes the overall ranking:** LoRA rank 16 correctly abstains on **61.97%** of unanswerable questions. Its advantage on this subset outweighs its smaller disadvantage on answerable questions.

These results compare *different adaptation scopes*, not just parameter counts: LoRA modifies Q/K/V projections with low-rank updates, whereas partial fine-tuning updates original parameters in selected Transformer blocks.

## Evaluation results

### Aggregate Exact Match and F1

![Grouped horizontal bar chart comparing overall Exact Match and token F1 for five SQuAD fine-tuning checkpoints](assets/overall_evaluation.png)

| Fine-tuning method | Overall EM | Overall F1 | Answerable EM | Answerable F1 | Unanswerable EM / F1 |
|:--|--:|--:|--:|--:|--:|
| LoRA · rank 8 | 57.57% | 64.55% | 56.97% | 65.48% | 60.30% |
| **LoRA · rank 16** | **60.20%** | **67.16%** | 59.81% | 68.30% | 61.97% |
| LoRA · rank 32 | 60.03% | 66.86% | 60.63% | 68.95% | 57.33% |
| Partial FT · freeze first 8 | 53.87% | 59.62% | 50.55% | 57.56% | **69.02%** |
| Partial FT · freeze first 16 | 57.73% | 64.87% | **62.33%** | **71.04%** | 36.73% |

**Evaluation population:** 3,000 total questions; all five checkpoints use the same 2,461 answerable / 539 unanswerable split. All percentages above are measured, not estimated from training loss.

The difference between **LoRA rank 16 and rank 32 is only five exact matches out of 3,000** (1,806 versus 1,801). The available data do not establish a robust advantage of rank 16 over rank 32 across new samples or training seeds. Conversely, rank 8 is meaningfully lower on the measured aggregate F1 score, and its lower answerable F1 indicates less effective answer generation in this experiment.

### Answerable vs. unanswerable questions

![Grouped horizontal bars contrasting exact-match accuracy on answerable and unanswerable questions](assets/answerability_evaluation.png)

**Answerable questions (n = 2,461).** These cases reward producing the reference answer, not merely returning a response. Partial FT with the first 16 layers frozen achieves the strongest score (62.33% EM, 71.04% F1). LoRA rank 32 is next on answerable EM (60.63%), followed by rank 16 (59.81%).

**Unanswerable questions (n = 539).** These cases reward correctly producing no answer. Partial FT with the first 8 layers frozen has the highest score (69.02%), while partial FT with the first 16 layers frozen has the lowest (36.73%). LoRA rank 16 sits between them at 61.97%. The weak unanswerable performance of the freeze-16 model offsets its stronger answer extraction.

For an empty ground-truth answer, the metric gives **EM = F1 = 1** if the normalized prediction is empty and **EM = F1 = 0** otherwise. Accordingly, *unanswerable EM and F1 are identical* in the reported evaluation. This is a measure of **correct empty responses**, not a separately calibrated confidence score.

### What the score differences mean

Consider **LoRA rank 16** against **partial FT freezing the first 16 layers**, evaluated on identical questions:

| Group | LoRA r=16 | Partial FT (freeze 16) | Difference for LoRA |
|:--|--:|--:|--:|
| Answerable, exact matches / 2,461 | 1,472 | 1,534 | −62 |
| Unanswerable, correct empty responses / 539 | 334 | 198 | +136 |
| **All examples, exact matches / 3,000** | **1,806** | **1,732** | **+74** |

Thus, the rank-16 LoRA adapter answers **62 fewer answerable questions exactly**, but correctly declines **136 more unanswerable questions**, yielding **74 more correct predictions in total**. Its overall EM advantage is **2.47 percentage points** (60.20% vs. 57.73%).

This is an *answerability trade-off*, not proof that one adaptation method intrinsically understands when to abstain. For example, a checkpoint can look good on unanswerable examples simply by being more inclined to terminate early. To test that possibility, the evaluator would need to measure **false abstention on answerable questions** separately from **wrong nonempty answers**.

The evaluation-set composition also matters. These aggregate scores weight answerable questions about **4.57×** as heavily as unanswerable questions because of their frequencies; a different mixture could change the ranking. The earlier 1,000-example evaluation happened to contain only answerable cases, explaining why its ranking differed from the expanded 3,000-example evaluation.

## Model and supervised QA objective

The QA system uses a **custom decoder-only Transformer trained from scratch**, rather than a downloaded instruction-tuned language model. Relevant architectural settings are:

| Component | Setting |
|:--|:--|
| Transformer blocks | 48 |
| Attention heads per block | 25 |
| Model/embedding width | 1,600 |
| Maximum context length | 1,024 tokens |
| Tokenization | GPT-2 BPE (`tiktoken`) |
| Padded vocabulary size | 50,304 |
| Output | Autoregressive answer text, terminated by the end-of-text token |

### Prompt representation

Each training example is serialized as:

```text
Context: <passage>
Question: <question>
Answer: <reference answer><|endoftext|>
```

Unanswerable SQuAD 2.0 questions are retained during preprocessing. If `answers['text']` is empty, the serializer emits an empty answer field:

```text
Context: <passage>
Question: <question with no supported answer>
Answer: <|endoftext|>
```

In the actual preprocessing code the answer begins after `"Answer: "`, so an empty answer may include a leading space token. The scoring code removes the end-of-text token and normalizes whitespace, ensuring that this is treated as an empty reference.

### Answer-only loss and block-causal masking

The supervised objective trains the model to **predict answer tokens**, including the terminating end-of-text token. Context and question tokens provide conditioning information but are ignored by the supervised cross-entropy loss:

$$
\mathcal{L}_{\mathrm{QA}} = -\frac{1}{|\mathcal{A}|}\sum_{t\in\mathcal{A}}\log p_{\theta}(x_t\mid x_{<t}),
$$

where $\mathcal{A}$ is the set of supervised answer/EOT target positions. Positions outside $\mathcal{A}$ use `ignore_index = -1`.

Because individual QA examples are packed together in fixed-length token windows, the dataloader also constructs a **block-causal attention mask**. A query can attend only to earlier positions **within the same `Context:` block**. This prevents one packed QA example from reading tokens from an unrelated example in the same training sequence. The implementation excludes incomplete terminal blocks from the supervised mask.

This differs from the usual unmodified causal mask: **autoregressive order is preserved, but cross-example attention is disallowed** during packed training.

## Fine-tuning methods

### Partial conventional fine-tuning

With `lora_rank = 0`, the original pretrained weights are updated in the unfrozen part of the network. The `n_frozen_layers` argument freezes the **first N Transformer blocks**, not the last N. This follows the trainer's implementation, which applies `requires_grad = False` to `transformer.layers[:n_frozen_layers]`.

| Training configuration | Frozen Transformer blocks | Trainable Transformer blocks |
|:--|:--|:--|
| `--lora_rank 0 --n_frozen_layers 8 --iter_batch_size 1` | First 8 | Remaining 40 |
| `--lora_rank 0 --n_frozen_layers 16 --iter_batch_size 1` | First 16 | Remaining 32 |

Parameters outside these blocks may remain trainable depending on their `requires_grad` settings. This is a partial-weight-update comparison against LoRA, not a parameter-matched experiment. In contrast to LoRA's Q/K/V-only updates, these configurations can update the unfrozen Transformer blocks' full attention and feed-forward weights.

### Low-Rank Adaptation (LoRA)

With `lora_rank > 0`, the pretrained matrices remain frozen and learnable low-rank factors are attached to the **query (Q), key (K), and value (V)** attention projections. Tested ranks: **8, 16, 32**.

For a pretrained projection $W$, the adapted projection is:

$$
W_{\mathrm{adapted}}=W+sBA,
$$

where $A$ and $B$ are trainable low-rank factors, $\mathrm{rank}(BA)\le r$, and $s$ denotes the implementation's adapter scaling.

At inference, the project's `apply_lora()` operation **merges** the learned $BA$ updates into the pretrained Q/K/V weights, so the model can generate without recomputing an adapter branch at each token.

This is a comparison of **LoRA on Q/K/V throughout the network** versus **direct weight updates in unfrozen Transformer blocks**. It is *not* an equal-parameter comparison, and the code/logs supplied do not establish exact parameter counts or peak VRAM for each configuration.

### Checkpoints

| Checkpoint | Adaptation |
|:--|:--|
| `lora_weights_8.pt` | LoRA Q/K/V, rank 8 |
| `lora_weights_16.pt` | LoRA Q/K/V, rank 16 |
| `lora_weights_32.pt` | LoRA Q/K/V, rank 32 |
| `best_model_nxl9_frozen_8.pt` | Partial FT, first 8 blocks frozen (40 unfrozen) |
| `best_model_nxl9_frozen_16.pt` | Partial FT, first 16 blocks frozen (32 unfrozen) |

## Dataset and evaluation protocol

### Tokenization, sharding, and splits

The preprocessing script uses Hugging Face Datasets to load:

```python
load_dataset("squad_v2", split="train")
```

It concatenates each context, question, reference answer, and `<|endoftext|>` marker, tokenizes using GPT-2 BPE, and writes consecutive shards of approximately **1,000,000 tokens** as `uint16` arrays. The dataset uses the following shard-level split:

| Split in this project | Selection |
|:--|:--|
| Validation | `all_shards[:1]` |
| Evaluation (`test`) | `all_shards[1:3]` |
| Training | `all_shards[3:]` |

The iterable test dataset reconstructs examples by finding `Context:` and `Answer:` boundaries and yielding a prompt plus the target answer. It does **not** remove empty-answer questions. The first 3,000 examples produced by the iterator were scored for this report, **in the original shard order**.

**Important:** This is a holdout of serialized *training-split tokens*, not the official SQuAD 2.0 held-out development set. Shard boundaries are based on token count, not document identity, so different shards can still share article or context material. The tokenizer also stores **only the first annotated valid answer**, not every reference accepted in official SQuAD scoring.

### Autoregressive evaluation and metrics

For each checkpoint, evaluation runs on one GPU with the same procedure:

1. Instantiate the same GPT architecture, load the partial-FT checkpoint or load pretrained base weights plus a LoRA adapter, then merge adapters if applicable.
2. Feed the prompt ending in `Answer:`; do **not** include ground truth in the input.
3. Decode greedily (`argmax`) until `<|endoftext|>` or the configured token limit (150 new tokens by default). Generation uses a sliding context bounded by 1,024 tokens.
4. Remove EOT from both prediction and reference; normalize by lowercasing, removing punctuation and `a/an/the`, then collapsing whitespace.
5. Calculate normalized **Exact Match (EM)** and word/token-overlap **F1**, both in aggregate and separately for answerable and unanswerable examples.

For nonempty references, F1 is the harmonic mean of unigram-token precision and recall. If either normalized prediction or reference is empty, F1 is **1 only if both are empty**; otherwise it is **0**.

All checkpoints were evaluated on the same **3,000 examples**; the evaluation output reports **0 prompts initially longer than the context window**.

## Distributed training and loss analysis

### Training configuration

Training was performed using **PyTorch Distributed Data Parallel (DDP) across four GPUs** with one model replica per rank. Gradient accumulation approximates a larger global batch while limiting per-device memory use. Gradients are synchronized on the final accumulation micro-step, rather than at every micro-step.

For fixed-length packed sequences, the nominal token count per optimizer update is:

$$
N_{\mathrm{tokens/update}}=\text{seq\_len}\times\text{local batch size}\times\text{GPU count}\times\text{accumulation steps}.
$$

This counts packed input positions, **not necessarily the number of supervised answer tokens** remaining after masking.

The supplied training log records **five 10,000-step experiments** in the exact execution order below. All curves contain **50 logged train/validation summaries**, sampled every **200 optimizer steps**. Training uses mixed-precision gradient scaling, a learning rate reaching approximately **5 × 10⁻⁵** around step 101, and cosine-style decay afterward. The log environment reports **PyTorch 2.8.0 with CUDA 12.6**.

| Experiment (in supplied log order) | Command-line overrides | Trainable Transformer blocks |
|:--|:--|:--|
| **LoRA r=32** | `--lora_rank 32 --n_frozen_layers 0` | Adapter weights only (Q/K/V across blocks) |
| **LoRA r=16** | `--lora_rank 16 --n_frozen_layers 0` | Adapter weights only (Q/K/V across blocks) |
| **LoRA r=8** | `--lora_rank 8 --n_frozen_layers 0` | Adapter weights only (Q/K/V across blocks) |
| **Partial FT · freeze first 8** | `--lora_rank 0 --n_frozen_layers 8 --iter_batch_size 1` | Last 40 blocks |
| **Partial FT · freeze first 16** | `--lora_rank 0 --n_frozen_layers 16 --iter_batch_size 1` | Last 32 blocks |

All plotted points are taken directly from the logged `avg train loss` and `avg val loss` summaries; curves are not smoothed or interpolated. The 50 validation summaries per configuration are periodic measurements of the validation objective, not the final 3,000-question autoregressive EM/F1 evaluation.

### Training-loss curves

![Training loss versus optimizer steps for LoRA ranks 32, 16, 8 and partial fine-tuning with 8 or 16 initial frozen blocks](assets/training_loss.png)

The three LoRA configurations have visibly different training-loss trajectories. At step 10,000, **LoRA r=32** ends at **0.0594**, **LoRA r=16** at **0.1656**, and **LoRA r=8** at **0.2716**. The higher-capacity LoRA ranks fit the supervised training targets more closely, but their final training losses do not predict the evaluation ranking: **r=16**, not r=32, has the best aggregate held-out EM/F1.

Both partial fine-tuning configurations approach near-zero training loss: **0.00058** when the **first 8 blocks are frozen** and **0.00093** when the **first 16 blocks are frozen**. With 40 or 32 Transformer blocks left trainable, their greater weight-update capacity can fit the training targets very closely. Neither near-zero value should be interpreted as evidence of excellent generalization.

A logarithmic y-axis makes all five curves readable on a shared figure without hiding the extremely small final losses of the partial FT configurations.

### Validation-loss curves

![Validation loss versus optimizer steps for named LoRA and partial fine-tuning configurations; dots identify minimum recorded validation losses](assets/validation_loss.png)

Each dot marks that configuration's **lowest logged validation loss**. The minimum is measured at the reporting cadence of 200 steps, so a lower loss may have occurred between reports.

| Configuration | Best logged validation step | Minimum val loss | Train loss at step 10,000 | Val loss at step 10,000 |
|:--|--:|--:|--:|--:|
| LoRA r=32 | 1,400 | 0.4700 | 0.0594 | 0.7012 |
| LoRA r=16 | 2,600 | 0.4781 | 0.1656 | 0.5421 |
| LoRA r=8 | 2,800 | 0.4851 | 0.2716 | 0.5195 |
| Partial FT · freeze first 8 | 400 | 0.5379 | 0.00058 | 1.2223 |
| Partial FT · freeze first 16 | 800 | **0.4693** | 0.00093 | 1.0998 |

The larger adaptation scope of partial fine-tuning corresponds to much faster fitting of the supervised answers, followed by substantially worse late-validation performance. The LoRA experiments maintain comparatively lower terminal validation losses, with **r=8** finishing at 0.5195 despite its higher training loss. **LoRA r=16** and **r=8** reach their logged validation minima later (steps 2,600–2,800) than the partial FT configurations (steps 400–800).

### Checkpoint selection and overfitting

The partial fine-tuning trajectories provide the clearest examples of **overfitting**:

- **Freeze first 8 / train remaining 40:** validation loss rises from **0.5379 at step 400** to **1.2223 at step 10,000**, even though training loss falls to **0.00058**.
- **Freeze first 16 / train remaining 32:** validation loss rises from **0.4693 at step 800** to **1.0998 at step 10,000**, with training loss just **0.00093**.

The latter run attains the **lowest single recorded validation loss** among the five experiments but loses much of that advantage by the end. Consequently, **best-validation checkpoint selection** is essential: selecting a last-step model would be especially misleading for these partial fine-tuning configurations.

LoRA r=32 also shows validation degradation after its minimum (0.4700 → 0.7012), though less dramatically than the two partial fine-tuning runs. LoRA r=8 and r=16 have smaller late-training validation gaps. These patterns are consistent with differences in adaptation capacity, but training configurations also differ in their batch-size overrides, so they are not a strictly controlled one-variable comparison.

**Connecting losses to QA behavior:** The best validation loss does not necessarily yield the best **EM/F1 or abstention behavior**. The three metrics measure different objectives: teacher-forced answer-token likelihood, autoregressive exact extraction, and ability to stop when an answer is unsupported. In the 3,000-example evaluation, LoRA r=16 leads aggregate EM/F1 even though partial FT with 16 frozen blocks records a slightly lower best validation loss. The comparison motivates reporting both training dynamics and final task behavior.

One log also reports an NCCL/CUDA out-of-memory warning during **process-group cleanup after final metrics were printed**; the 10,000-step summary remains present. The error belongs in raw training records but does not remove the logged loss data.

## Reproduction

**Training (4 GPUs):** The runs above use the following argument combinations, listed in execution order: `--lora_rank 32 --n_frozen_layers 0`, `--lora_rank 16 --n_frozen_layers 0`, `--lora_rank 8 --n_frozen_layers 0`, `--lora_rank 0 --n_frozen_layers 8 --iter_batch_size 1`, and `--lora_rank 0 --n_frozen_layers 16 --iter_batch_size 1`.

```bash
torchrun --standalone --nproc_per_node=4 \
  train_eval_squad.py --phase train --config config.json
```

**Evaluation (1 GPU):**

```bash
torchrun --standalone --nproc_per_node=1 \
  <evaluation-script>.py --config config.json
```

Replace `<evaluation-script>.py` with the standalone evaluation script used in this repository. It reads checkpoints from its configured checkpoint list and writes `qa_inference_results.json`. The reported comparison used `test_limit = 3000`; setting `test_limit` to `null` instead evaluates the available custom test iterator in full. The generation limit defaults to `test_max_new_tokens = 150` unless overridden.

The test code checks the expected model/adaptor keys, loads pretrained GPT weights before LoRA weights, and calls `apply_lora()` before inference. For partial-FT models it loads the fine-tuned model state directly.

To embed the plots in a GitHub README, retain the four PNGs in the repository's `assets/` directory alongside this file; Markdown image references use **relative paths**.
