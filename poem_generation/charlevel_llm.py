import os
import sys

import torch
import torch.nn as nn

from transformer import transformer


class CharLevelLLM(nn.Module):
    def __init__(self, dictionary_length, num_layers=16, seq_len=200, embed_dim=512, num_heads=16):
        super(CharLevelLLM, self).__init__()
        self.transformer = transformer.MultiLayerTransformer(num_layers=num_layers, embed_dim=embed_dim, num_heads=num_heads)
        self.embeddin_table = nn.Embedding(dictionary_length, embed_dim)
        self.pos_embed = nn.Embedding(seq_len, embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.ln_f = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, dictionary_length, bias=True)
        self.classifier.weight = self.embeddin_table.weight

    def forward(self, x):
        embedded_chars = self.embeddin_table(x) + self.pos_embed(torch.arange(x.shape[1], device=x.device))
        embedded_chars = self.dropout(embedded_chars)
        output = self.transformer(embedded_chars)
        output = self.ln_f(output)
        return self.classifier(output)