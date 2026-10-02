"""Top-k Mixture-of-Experts FFN and sparse upcycling of a dense GPT."""
import copy

import torch
import torch.nn as nn

from src.model import GPT, MLP


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


def upcycle(dense: GPT, n_experts: int = 4, top_k: int = 2,
            router_std: float = 0.02, seed: int = 0) -> GPT:
    """Sparse upcycling: copy each dense FFN into every expert, add a fresh router.

    Identical experts + gates that sum to 1 => output equals the dense FFN exactly,
    whatever the router picks. So the router is random (spreads load from step 0)
    and no noise is added to the experts (that would break function preservation).
    """
    model = copy.deepcopy(dense)
    g = torch.Generator().manual_seed(seed)
    for blk in model.blocks:
        dense_ffn = blk.ffn
        moe = MoE(model.cfg.d_model, model.cfg.d_ff, n_experts, top_k)
        for expert in moe.experts:
            expert.load_state_dict(dense_ffn.state_dict())
        with torch.no_grad():
            moe.router.weight.copy_(torch.randn(moe.router.weight.shape, generator=g) * router_std)
        blk.ffn = moe
    return model
