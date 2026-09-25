import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import Block
from .reversible import make_rev_blocks, rev_forward


def _ce_sum(h, t, head):
    return F.cross_entropy(head(h).float(), t, reduction="sum")


def chunked_lm_loss(h, targets, head, chunk_tokens):
    """Mean CE computed chunk-by-chunk; each chunk's logits are recomputed in backward,
    so peak logits memory is O(chunk_tokens * vocab) instead of O(B * T * vocab)."""
    h = h.reshape(-1, h.size(-1))
    t = targets.reshape(-1)
    total = h.new_zeros((), dtype=torch.float32)
    for i in range(0, h.size(0), chunk_tokens):
        total = total + checkpoint(_ce_sum, h[i:i + chunk_tokens], t[i:i + chunk_tokens], head, use_reentrant=False)
    result = total / t.numel()
    if h.dtype == torch.float64:
        result = result.to(h.dtype)
    return result


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        if cfg.arch == "baseline":
            self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        else:
            self.blocks = make_rev_blocks(cfg)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied embeddings
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def embed(self, idx):
        return self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))

    def head(self, x):
        return self.lm_head(self.ln_f(x))

    def forward(self, idx, targets=None, use_rev_fn=True):
        x = self.embed(idx)
        if self.cfg.arch == "baseline":
            for blk in self.blocks:
                x = blk(x)
        else:
            x = rev_forward(self.blocks, x, self.cfg.arch, memory_saving=use_rev_fn)
        if targets is None:
            return self.head(x), None
        if self.cfg.chunked_loss:
            return None, chunked_lm_loss(x, targets, self.head, self.cfg.loss_chunk_tokens)
        logits = self.head(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        # Preserve fp64 precision (mirrors chunked_lm_loss), but never downcast to fp16 under autocast.
        if logits.dtype == torch.float64:
            loss = loss.to(logits.dtype)
        return logits, loss
