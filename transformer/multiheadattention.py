import torch
import torch.nn as nn
import torch.nn.functional as F


# Causal Multi Head Attention
class CMHAttention(nn.Module):
    def __init__(self,
                 embed_dim=512,
                 seq_len=1024,
                 mlp_ratio=2.0,
                 num_heads=16,
                 dropout=0.1,
                 use_flash=True,
                ):
        super(CMHAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.mlp_ratio = int(mlp_ratio)
        
        self.k = nn.Linear(self.embed_dim, self.embed_dim)
        self.q = nn.Linear(self.embed_dim, self.embed_dim)
        self.v = nn.Linear(self.embed_dim, self.embed_dim)
        
        attention_mask = torch.tril(torch.ones(1, self.num_heads, seq_len, seq_len, dtype=torch.bool))
        self.register_buffer("attention_mask", attention_mask)
        
        self.ln1 = nn.LayerNorm(self.embed_dim)
        self.ln2 = nn.LayerNorm(self.embed_dim)
        self.use_flash = use_flash

        self.fnn = nn.Sequential(
            nn.Linear(self.embed_dim, self.mlp_ratio * self.embed_dim),
            nn.GELU(),
            nn.Linear(self.mlp_ratio * self.embed_dim, self.embed_dim),
            nn.Dropout(p=dropout),
        )

        self.project = nn.Linear(self.embed_dim, self.embed_dim)


    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        
        C_heads = C//self.num_heads

        x_norm = self.ln1(x)

        # split into kqv and split into multi heads
        k = self.k(x_norm).reshape(B, N, self.num_heads, C_heads).transpose(-2, -3) # (B, num_heads, N, C_heads)
        q = self.q(x_norm).reshape(B, N, self.num_heads, C_heads).transpose(-2, -3) # (B, num_heads, N, C_heads)
        v = self.v(x_norm).reshape(B, N, self.num_heads, C_heads).transpose(-2, -3) # (B, num_heads, N, C_heads)

        if attention_mask is not None:
            self.attention_mask = attention_mask

        if self.use_flash:
            combined_tokens = F.scaled_dot_product_attention(q, k, v,
                                                             attn_mask=active_mask,
                                                             dropout_p=0.0 if not self.training else 0.1,
                                                             is_causal=False,
                                                            )

        else:
            # # compute attention matrix
            attention_matrix = (q @ k.transpose(-1, -2))/(C_heads**0.5)   # (B, num_heads, N, N)

            # # scale the sum of attention weights for a token to 1
            attention_matrix = attention_matrix.masked_fill(~attention_mask, float("-inf"))
            attention_matrix = torch.softmax(attention_matrix, dim=-1) # (B, num_heads, N, N)

            # create the weigted some of token representations
            combined_tokens = attention_matrix @ v # (B, num_heads, N, C_heads)

        # reshape back to original representations
        combined_tokens = combined_tokens.transpose(-2,-3).reshape(B, N, C)  # (B, N, C)

        # project back to the original embedding space
        combined_tokens = self.project(combined_tokens)  # (B, N, C)
        combined_tokens = x + combined_tokens  # (B, N, C)

        # pass from fnn
        mlp_output = self.fnn(self.ln2(combined_tokens))  # (B, N, C)
        out = combined_tokens + mlp_output

        return out
