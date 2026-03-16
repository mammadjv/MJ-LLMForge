import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .multiheadattention import CMHAttention


class MultiLayerTransformer(nn.Module):
    def __init__(self, seq_len, num_layers=10, embed_dim=512, num_heads=16, dropout=0.2, use_checkpointing=False, use_flash=True):
        super(MultiLayerTransformer, self).__init__()
        
        self.layers = nn.ModuleList()
        self.use_checkpointing = use_checkpointing
        for _ in range(num_layers):
            self.layers.append(CMHAttention(embed_dim=embed_dim, dropout=dropout, num_heads=num_heads, seq_len=seq_len, use_flash=use_flash))
            
    def forward(self, x):
        for layer in self.layers:
            if self.use_checkpointing:
                x = checkpoint(layer, x, use_reentrant=False, preserve_rng_state=True)

            else:
                x = layer(x)

        return x

    
