"""Adam, written out by hand.

Note on eps placement: PyTorch computes
    denom = sqrt(v_t) / sqrt(1 - b2**t) + eps
    w    -= lr / (1 - b1**t) * m_t / denom
which equals  w -= lr * m_hat / (sqrt(v_hat) + eps).
eps sits OUTSIDE the sqrt. Writing sqrt(v_hat + eps) disagrees in the 8th decimal.
"""
import math
from typing import Iterable

import torch


def adam_step_scalar(w, g, m, v, t, lr, b1, b2, eps, bias_correction=True):
    """One Adam step on a single scalar weight. Returns the full trace."""
    m = b1 * m + (1.0 - b1) * g
    v = b2 * v + (1.0 - b2) * g * g
    if bias_correction:
        m_hat = m / (1.0 - b1 ** t)
        v_hat = v / (1.0 - b2 ** t)
    else:
        m_hat, v_hat = m, v
    update = -lr * m_hat / (math.sqrt(v_hat) + eps)
    return {
        "t": t, "g": g, "m": m, "v": v,
        "m_hat": m_hat, "v_hat": v_hat,
        "update": update, "w_new": w + update,
    }


class ManualAdam:
    """Minimal Adam. No weight decay, no amsgrad - matches torch.optim.Adam defaults."""

    def __init__(self, params: Iterable[torch.Tensor], lr=1e-3,
                 betas=(0.9, 0.999), eps=1e-8, bias_correction=True):
        self.params = list(params)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.bias_correction = bias_correction
        self.t = 0
        self.state = {
            p: {"m": torch.zeros_like(p), "v": torch.zeros_like(p)}
            for p in self.params
        }

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            if set_to_none:
                p.grad = None
            elif p.grad is not None:
                p.grad.zero_()

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for p in self.params:
            if p.grad is None:
                continue
            g = p.grad
            s = self.state[p]
            s["m"].mul_(self.b1).add_(g, alpha=1.0 - self.b1)
            s["v"].mul_(self.b2).addcmul_(g, g, value=1.0 - self.b2)
            if self.bias_correction:
                m_hat = s["m"] / bc1
                v_hat = s["v"] / bc2
            else:
                m_hat, v_hat = s["m"], s["v"]
            p.add_(-self.lr * m_hat / (v_hat.sqrt() + self.eps))
