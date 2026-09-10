"""LR schedules. Every one returns a plain float given an integer step."""
import math


def linear_warmup(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def cosine_with_warmup(step, peak_lr, warmup_steps, total_steps, min_lr_frac=0.1):
    """Linear warmup then cosine decay to min_lr_frac * peak_lr at total_steps.

    Key property for Q4: this curve is defined by total_steps. Truncating the run
    early does NOT give you the annealed endpoint - you land mid-decay.
    """
    if step < warmup_steps:
        return peak_lr * linear_warmup(step, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_frac + (1.0 - min_lr_frac) * cos)


def wsd(step, peak_lr, warmup_steps, decay_start, total_steps, min_lr_frac=0.0):
    """Warmup -> Stable -> Decay, with a linear (1 - x) decay phase.

    Key property for Q4: the stable phase is horizon-agnostic. You can decide at
    any point to start the decay and land a usable checkpoint a few steps later.
    """
    if step < warmup_steps:
        return peak_lr * linear_warmup(step, warmup_steps)
    if step < decay_start:
        return peak_lr
    progress = (step - decay_start) / max(1, total_steps - decay_start)
    progress = min(1.0, progress)
    return peak_lr * (min_lr_frac + (1.0 - min_lr_frac) * (1.0 - progress))
