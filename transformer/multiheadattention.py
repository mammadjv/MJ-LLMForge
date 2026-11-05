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
                 lora_rank=0,
                 lora_alpha=16,
                 attention_dropout=0.0,
                ):
        super(CMHAttention, self).__init__()

        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.mlp_ratio = int(mlp_ratio)
        self.attention_dropout = attention_dropout

        self.lora_rank = lora_rank

        self.k = nn.Linear(self.embed_dim, self.embed_dim)
        self.q = nn.Linear(self.embed_dim, self.embed_dim)
        self.v = nn.Linear(self.embed_dim, self.embed_dim)

        self.head_dim = embed_dim // num_heads

        # Learnable RMSNorm gains: [1, H, 1, head_dim]
        self.q_gamma = nn.Parameter(
            torch.ones(1, num_heads, 1, self.head_dim)
        )
        self.k_gamma = nn.Parameter(
            torch.ones(1, num_heads, 1, self.head_dim)
        )

        attention_mask = torch.ones(1, 1, self.seq_len, self.seq_len, dtype=torch.bool).tril()
        self.register_buffer("attention_mask", attention_mask, persistent=False)

        if self.lora_rank > 0:
            self.lora_scale = 2
            self.Bk = nn.Parameter(torch.zeros(self.embed_dim, self.lora_rank))
            self.Ak = nn.Parameter(torch.randn(self.lora_rank, self.embed_dim)*0.1)

            self.Bq = nn.Parameter(torch.zeros(self.embed_dim, self.lora_rank))
            self.Aq = nn.Parameter(torch.randn(self.lora_rank, self.embed_dim)*0.1)

            self.Bv = nn.Parameter(torch.zeros(self.embed_dim, self.lora_rank))
            self.Av = nn.Parameter(torch.randn(self.lora_rank, self.embed_dim)*0.1)

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
        self.lora_merged = False

    @staticmethod
    def qk_rms_norm(x, gamma, eps=1e-6):
        dtype = x.dtype
        x = x.float()
    
        x = x * torch.rsqrt(
            x.square().mean(dim=-1, keepdim=True) + eps
        )
    
        return (x * gamma.float()).to(dtype)

    def apply_lora(self):
        with torch.no_grad():
            self.q.weight += self.lora_scale * (self.Bq @ self.Aq)
            self.k.weight += self.lora_scale * (self.Bk @ self.Ak)
            self.v.weight += self.lora_scale * (self.Bv @ self.Av)
            self.lora_merged = True

    def forward(self, x, attention_mask=None):
        B, N, C = x.shape
        
        C_heads = C//self.num_heads

        x_norm = self.ln1(x)

        # split into kqv and split into multi heads
        k = self.k(x_norm).reshape(B, N, self.num_heads, C_heads).transpose(-2, -3) # (B, num_heads, N, C_heads)
        q = self.q(x_norm).reshape(B, N, self.num_heads, C_heads).transpose(-2, -3) # (B, num_heads, N, C_heads)
        v = self.v(x_norm).reshape(B, N, self.num_heads, C_heads).transpose(-2, -3) # (B, num_heads, N, C_heads)

        if self.lora_rank > 0 and not self.lora_merged:
            k_lora = F.linear(
                F.linear(x_norm, self.Ak),
                self.Bk,
            ).reshape(B, N, self.num_heads, C_heads).transpose(-2, -3) # (B, num_heads, N, C_heads)

            q_lora = F.linear(
                F.linear(x_norm, self.Aq),
                self.Bq,
            ).reshape(B, N, self.num_heads, C_heads).transpose(-2, -3) # (B, num_heads, N, C_heads)

            v_lora = F.linear(
                F.linear(x_norm, self.Av),
                self.Bv,
            ).reshape(B, N, self.num_heads, C_heads).transpose(-2, -3) # (B, num_heads, N, C_heads)

            q = q + self.lora_scale * q_lora
            k = k + self.lora_scale * k_lora
            v = v + self.lora_scale * v_lora

        # QK RMSNorm with trainable gamma
        q = self.qk_rms_norm(q, self.q_gamma)
        k = self.qk_rms_norm(k, self.k_gamma)

        active_mask = attention_mask if attention_mask is not None else self.attention_mask

        if active_mask.size(-1) > N:
            active_mask = active_mask[..., :N, :N]

        if self.use_flash:
            combined_tokens = F.scaled_dot_product_attention(q, k, v,
                                                             attn_mask=active_mask,
                                                             dropout_p=0.0 if not self.training else self.attention_dropout,
                                                             is_causal=False,
                                                            )

        else:
            attention_matrix = (q @ k.transpose(-1, -2)) / (C_heads ** 0.5)   # (B, num_heads, N, N)
            attention_matrix = attention_matrix.masked_fill(~active_mask, float("-inf"))

            attention_matrix = torch.softmax(attention_matrix, dim=-1) # (B, num_heads, N, N)
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
