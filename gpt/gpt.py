import os
import sys

import torch
import torch.nn as nn

from transformer import transformer


class GPT(nn.Module):
    def __init__(self, dictionary_length, seq_len, num_layers=16, embed_dim=512, num_heads=16, use_checkpointing=False, use_flash=True):
        super(GPT, self).__init__()
        self.transformer = transformer.MultiLayerTransformer(seq_len=seq_len,
                                                             num_layers=num_layers,
                                                             embed_dim=embed_dim,
                                                             num_heads=num_heads,
                                                             use_checkpointing=use_checkpointing,
                                                             use_flash=use_flash,
                                                            )
        self.embeddin_table = nn.Embedding(dictionary_length, embed_dim)
        self.pos_embed = nn.Embedding(seq_len, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, dictionary_length, bias=True)
        self.classifier.weight = self.embeddin_table.weight

    def forward(self, x, causal_mask):
        embedded_chars = self.embeddin_table(x) + self.pos_embed(torch.arange(x.shape[1], device=x.device))
        embedded_chars = self.dropout(embedded_chars)
        output = self.transformer(embedded_chars, causal_mask)
        output = self.ln_f(output)
        return self.classifier(output)