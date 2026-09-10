"""Per-layer update-to-weight ratio: ||dw||_RMS / ||w||_RMS.

Rule of thumb from the lecture: healthy pretraining sits near 1e-3.
Much higher and the step is destabilising; much lower and the layer is frozen.
RMS rather than raw L2 so tensors of different sizes are comparable.
"""
from typing import Dict, List

import torch


def _rms(t: torch.Tensor) -> float:
    return t.detach().float().pow(2).mean().sqrt().item()


class UpdateRatioLogger:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self._prev: Dict[str, torch.Tensor] = {}
        self.rows: List[dict] = []

    @torch.no_grad()
    def snapshot(self) -> None:
        """Call immediately BEFORE optimizer.step()."""
        self._prev = {
            n: p.detach().clone()
            for n, p in self.model.named_parameters() if p.requires_grad
        }

    @torch.no_grad()
    def record(self, step: int, lr: float) -> None:
        """Call immediately AFTER optimizer.step()."""
        for n, p in self.model.named_parameters():
            if not p.requires_grad or n not in self._prev:
                continue
            delta = p.detach() - self._prev[n]
            w_rms = _rms(self._prev[n])
            d_rms = _rms(delta)
            self.rows.append({
                "step": step, "layer": n, "lr": lr,
                "w_rms": w_rms, "d_rms": d_rms,
                "ratio": (d_rms / w_rms) if w_rms > 0 else 0.0,
            })
