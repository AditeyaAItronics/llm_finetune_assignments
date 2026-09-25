import gc

import torch

from .config import ModelConfig
from .model import GPT
from .train import amp_dtype


def _is_oom(e: Exception) -> bool:
    """True for torch.cuda.OutOfMemoryError, or a RuntimeError whose message names an
    out-of-memory condition (e.g. cudaMalloc / cuDNN 'ALLOC_FAILED')."""
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(e, RuntimeError):
        msg = str(e).lower()
        return "out of memory" in msg or "alloc_failed" in msg
    return False


def fits(mc: ModelConfig, batch_size: int, steps: int = 2, use_rev_fn: bool = True) -> bool:
    """True if `steps` full train steps (fwd + bwd + AdamW) run at this batch size without OOM."""
    model = opt = None
    try:
        model = GPT(mc).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        for _ in range(steps):
            x = torch.randint(mc.vocab_size, (batch_size, mc.block_size), device="cuda")
            with torch.autocast("cuda", dtype=amp_dtype()):
                _, loss = model(x, x, use_rev_fn=use_rev_fn)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return True
    except RuntimeError as e:
        if _is_oom(e):
            return False
        raise
    finally:
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()


def find_max_batch(mc: ModelConfig, start: int = 8, limit: int = 4096, safety: float = 0.9,
                    use_rev_fn: bool = True) -> dict:
    """Double until OOM, then binary-search the exact boundary."""
    lo, hi = 0, start
    while hi <= limit and fits(mc, hi, use_rev_fn=use_rev_fn):
        lo, hi = hi, hi * 2
    hi = min(hi, limit + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mc, mid, use_rev_fn=use_rev_fn):
            lo = mid
        else:
            hi = mid
    if lo == 0:
        raise RuntimeError(f"batch_size=1 does not fit for arch={mc.arch}")
    recommended = int(lo * safety) // 8 * 8 or max(1, int(lo * safety))
    return {"arch": mc.arch, "max_batch": lo, "recommended": recommended,
            "first_oom": hi if hi <= limit else None, "use_rev_fn": use_rev_fn}
