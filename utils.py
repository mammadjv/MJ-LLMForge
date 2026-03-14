import torch.nn as nn
import torch


def weights_init(m):
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LayerNorm):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def get_regularization_parameters(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if name.endswith("project.weight"):
            param.data *= 1 / (2 * num_layers) ** 0.5
    
        if not param.requires_grad:
            continue

        if param.ndim == 1 or "embed" in name or "bias" in name:
            no_decay.append(param)

        else:
            decay.append(param)

    return decay, no_decay


def linear_lr_update(optimizer, e, base_lr, lr_reduction_epoch=3):
    for param in optimizer.param_groups:
        param['lr'] = base_lr * (0.5 ** ((e - 1) // lr_reduction_epoch))

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