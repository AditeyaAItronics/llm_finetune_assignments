"""Reversible residual stacks.

Both variants keep two residual streams (x1, x2) and are exactly invertible, so the
backward pass can rebuild every layer's input from its output instead of storing it.

euler    - additive coupling (RevNet / Reformer), a symplectic-Euler step:
           y1 = x1 + h f(x2);  y2 = x2 + h g(y1)
midpoint - leapfrog / explicit-midpoint step on states (x_{n-1}, x_n):
           (a, b) -> (b, a + 2h F(b)),  F = full pre-LN transformer delta
"""
import torch
import torch.nn as nn

from .layers import AttnBranch, Block, MLPBranch


class EulerRevBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.f = AttnBranch(cfg)
        self.g = MLPBranch(cfg)
        self.h = cfg.h

    def forward(self, x1, x2):
        y1 = x1 + self.h * self.f(x2)
        y2 = x2 + self.h * self.g(y1)
        return y1, y2

    def inverse(self, y1, y2):
        x2 = y2 - self.h * self.g(y1)
        x1 = y1 - self.h * self.f(x2)
        return x1, x2


class MidpointRevBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block = Block(cfg)
        self.h = cfg.h

    def forward(self, a, b):
        return b, a + 2 * self.h * self.block.delta(b)

    def inverse(self, y1, y2):
        b = y1
        return y2 - 2 * self.h * self.block.delta(b), b


def make_rev_blocks(cfg):
    cls = {"euler": EulerRevBlock, "midpoint": MidpointRevBlock}[cfg.arch]
    return nn.ModuleList(cls(cfg) for _ in range(cfg.n_layer))


class _RevStack(torch.autograd.Function):
    """Runs all blocks without storing activations; backward reconstructs them layer by layer."""

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x1, x2, blocks, *params):
        ctx.blocks = blocks
        with torch.no_grad():
            for blk in blocks:
                x1, x2 = blk(x1, x2)
        ctx.save_for_backward(x1, x2)  # only the final two streams are kept
        return x1, x2

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, dy1, dy2):
        y1, y2 = ctx.saved_tensors
        per_block = []
        for blk in reversed(ctx.blocks):
            with torch.no_grad():
                x1, x2 = blk.inverse(y1, y2)
            # Drop cached autocast weight casts made under no_grad so the recompute below
            # casts fresh, instead of silently reusing (possibly stale) no_grad casts.
            torch.clear_autocast_cache()
            with torch.enable_grad():
                x1r = x1.detach().requires_grad_(True)
                x2r = x2.detach().requires_grad_(True)
                o1, o2 = blk(x1r, x2r)
                ps = tuple(blk.parameters())
                grads = torch.autograd.grad((o1, o2), (x1r, x2r) + ps, (dy1, dy2), allow_unused=True)
            dy1, dy2 = grads[0], grads[1]
            per_block.append([g if g is not None else torch.zeros_like(p) for g, p in zip(grads[2:], ps)])
            y1, y2 = x1, x2
        flat = [g for block_grads in reversed(per_block) for g in block_grads]
        return (dy1, dy2, None, *flat)


def rev_forward(blocks, x, arch, memory_saving=True):
    x1, x2 = x, x
    if memory_saving:
        params = [p for blk in blocks for p in blk.parameters()]
        x1, x2 = _RevStack.apply(x1, x2, blocks, *params)
    else:
        for blk in blocks:
            x1, x2 = blk(x1, x2)
    return (x1 + x2) / 2 if arch == "euler" else x2


@torch.no_grad()
def _reconstruction_stats(blocks, x):
    """Forward through all blocks, then invert back down the stack (as backward does).
    Returns (max abs error, max abs stream value) over all stored states."""
    states = [(x, x)]
    for blk in blocks:
        states.append(blk(*states[-1]))
    y1, y2 = states[-1]
    err = 0.0
    scale = 0.0
    for i in range(len(blocks) - 1, -1, -1):
        y1, y2 = blocks[i].inverse(y1, y2)
        err = max(err, (y1 - states[i][0]).abs().max().item(), (y2 - states[i][1]).abs().max().item())
        scale = max(scale, states[i][0].abs().max().item(), states[i][1].abs().max().item())
    return err, scale


def reconstruction_error(blocks, x) -> float:
    """Max |true input - reconstructed input| over all layers."""
    err, _ = _reconstruction_stats(blocks, x)
    return err


def reconstruction_error_rel(blocks, x) -> float:
    """Max abs reconstruction error, relative to the max |stream| over all stored states."""
    err, scale = _reconstruction_stats(blocks, x)
    return err / scale if scale > 0 else err


def grad_agreement(model, x, y) -> dict:
    """Cosine and relative L2 error between block-param grads from the memory-saving
    backward and plain autograd on the same batch (same autocast context as the caller)."""
    block_params = [p for p in model.blocks.parameters()]
    _, loss_plain = model(x, y, use_rev_fn=False)
    g_plain = torch.autograd.grad(loss_plain, block_params, retain_graph=False, allow_unused=True)
    g_plain = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_plain, block_params)]

    _, loss_rev = model(x, y, use_rev_fn=True)
    g_rev = torch.autograd.grad(loss_rev, block_params, retain_graph=False, allow_unused=True)
    g_rev = [g if g is not None else torch.zeros_like(p) for g, p in zip(g_rev, block_params)]

    flat_plain = torch.cat([g.reshape(-1).float() for g in g_plain])
    flat_rev = torch.cat([g.reshape(-1).float() for g in g_rev])
    cos = torch.nn.functional.cosine_similarity(flat_plain, flat_rev, dim=0).item()
    denom = flat_plain.norm().item()
    rel_err = (flat_rev - flat_plain).norm().item() / denom if denom > 0 else (flat_rev - flat_plain).norm().item()
    return {"grad_cos": cos, "grad_rel_err": rel_err}
