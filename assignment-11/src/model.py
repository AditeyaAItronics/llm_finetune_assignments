"""A tiny decoder-only transformer whose only knob is `width`.

Depth, heads, seq_len and vocab are held fixed across the width sweep so that
Q5's LR-vs-width curve isolates width.

KNOWN CONFOUND, named in the README: with n_heads fixed at 4, head_dim scales
with width (64 -> 128 -> 256), so the attention logit scale 1/sqrt(head_dim)
moves along with width. Fixing head_dim and scaling n_heads instead is the other
defensible convention; muP fixes head_dim. Whichever you pick, one variable rides
along with width, so state which one you chose and that lr* absorbs it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, width: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, n_heads, batch_first=True, bias=False)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width, bias=False),
            nn.GELU(),
            nn.Linear(4 * width, width, bias=False),
        )

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, width: int, n_layers: int = 2,
                 n_heads: int = 4, seq_len: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.tok = nn.Embedding(vocab_size, width)
        self.pos = nn.Embedding(seq_len, width)
        self.blocks = nn.ModuleList([Block(width, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.apply(self._init)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    @staticmethod
    def _init(m):
        # Standard parameterisation: std = 1/sqrt(fan_in). Q5 measures how the
        # optimal LR drifts with width UNDER THIS CHOICE. (muP would remove the drift.)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.MultiheadAttention):
            # nn.MultiheadAttention is not an nn.Linear, so `apply` skips its
            # internal projections. Init them explicitly or they keep torch's
            # xavier default, which is NOT the fan-in rule the rest of the model
            # uses - and Q5's whole premise is a consistent parameterisation.
            if m.in_proj_weight is not None:
                nn.init.normal_(m.in_proj_weight, mean=0.0,
                                std=1.0 / math.sqrt(m.embed_dim))
            nn.init.normal_(m.out_proj.weight, mean=0.0,
                            std=1.0 / math.sqrt(m.embed_dim))

    def forward(self, x, y=None):
        b, t = x.shape
        pos = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(pos)[None]
        m = self.mask[:t, :t]
        for blk in self.blocks:
            h = blk(h, m)
        logits = self.head(self.ln_f(h))
        if y is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        return logits, loss
