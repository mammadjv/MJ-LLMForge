import math

import torch.nn as nn
import torch
import bitsandbytes as bnb

def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


@torch.no_grad()
def projection_scaling(model, num_layers):
    scale = (2 * num_layers) ** -0.5

    for name, param in model.named_parameters():
        if not name.startswith("transformer.layers."):
            continue

        if name.endswith((
            ".project.weight",  # Attention output
            ".fnn.2.weight",    # MLP output
        )):
            param.mul_(scale)


def get_regularization_parameters(model, num_layers):
    decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        param_name = name.rsplit(".", 1)[-1]

        if (
            param.ndim <= 1
            or "embed" in name
            or param_name.endswith("bias")
            or param_name in {"q_gamma", "k_gamma"}
        ):
            no_decay.append(param)
        else:
            decay.append(param)

    return decay, no_decay


def linear_lr_update(optimizer, e, base_lr, lr_reduction_epoch=3):
    for param in optimizer.param_groups:
        param['lr'] = base_lr * (0.5 ** ((e - 1) // lr_reduction_epoch))

    return optimizer


def cosine_lr_update(optimizer, step, base_lr, max_steps, max_warmup_steps, min_lr=2e-5):
    if step >= max_steps:
        lr = min_lr

    else:
        progress = (step - max_warmup_steps) / (
            max_steps - max_warmup_steps
        )
        progress = max(0.0, min(1.0, progress))

        lr = min_lr + 0.5 * (base_lr - min_lr) * (
            1.0 + math.cos(math.pi * progress)
        )

    for group in optimizer.param_groups:
        group["lr"] = lr

    return optimizer


def optimizer_warmup(optimizer, base_lr, step, max_warmup_steps=5000):
    for param in optimizer.param_groups:
        param['lr'] = (base_lr * step)/max_warmup_steps

    return optimizer


def top_k_sampling(logits, k=10, temperature=0.8):
    # 1. Get the logits for the LAST token only: shape becomes [batch, vocab_size]
    logits = logits[:, -1, :]/temperature

    # 2. Now perform Top-K on a 2D tensor
    v, _ = torch.topk(logits, k=k)
    min_values = v[:, -1:] # Keep the last dimension for broadcasting

    # 3. Apply the mask
    mask = logits < min_values
    logits[mask] = float('-inf')

    # 4. Sample
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def replace_with_4bit(module):
    for name, child in list(module.named_children()):
        if isinstance(child, bnb.nn.Linear4bit):
            continue

        if type(child) is nn.Linear:
            if any(p.requires_grad for p in child.parameters()):
                continue

            new_layer = bnb.nn.Linear4bit(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                compute_dtype=torch.float16,
                quant_type="nf4",
                compress_statistics=True,
            )

            new_layer.load_state_dict(child.state_dict())
            new_layer.requires_grad_(False)
            setattr(module, name, new_layer)

        else:
            replace_with_4bit(child)