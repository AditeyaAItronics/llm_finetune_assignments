"""Top-k Mixture-of-Experts FFN and sparse upcycling of a dense GPT."""
import torch
import torch.nn as nn

from src.model import MLP


class MoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([MLP(d_model, d_ff) for _ in range(n_experts)])
        self.aux_loss = torch.zeros(())
        self.last_frac = torch.full((n_experts,), 1.0 / n_experts)

    def forward(self, x):
        b, t, d = x.shape
        h = x.reshape(-1, d)
        probs = self.router(h).softmax(dim=-1)                 # [N, E]
        top_p, top_i = probs.topk(self.top_k, dim=-1)          # [N, k]
        gates = top_p / top_p.sum(dim=-1, keepdim=True)        # renormalise: sums to 1
        out = torch.zeros_like(h)
        for e, expert in enumerate(self.experts):
            tok, slot = (top_i == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            out.index_add_(0, tok, gates[tok, slot, None] * expert(h[tok]))
        # Switch Transformer load-balancing loss: E * sum_e f_e * P_e (=1 when balanced)
        counts = torch.bincount(top_i.flatten(), minlength=self.n_experts).float()
        frac = counts / counts.sum()
        self.aux_loss = self.n_experts * (frac * probs.mean(dim=0)).sum()
        self.last_frac = frac.detach()
        return out.reshape(b, t, d)
