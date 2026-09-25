import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class AttnBranch(nn.Module):
    """f(x) = Attn(LN(x))"""
    def __init__(self, cfg):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)

    def forward(self, x):
        return self.attn(self.ln(x))


class MLPBranch(nn.Module):
    """g(x) = MLP(LN(x))"""
    def __init__(self, cfg):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        return self.mlp(self.ln(x))


class Block(nn.Module):
    """Standard pre-LN transformer block: x + F(x)."""
    def __init__(self, cfg):
        super().__init__()
        self.f = AttnBranch(cfg)
        self.g = MLPBranch(cfg)

    def delta(self, x):
        a = self.f(x)
        return a + self.g(x + a)

    def forward(self, x):
        return x + self.delta(x)
