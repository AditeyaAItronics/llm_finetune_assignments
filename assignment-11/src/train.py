"""One training loop, shared by Q3, Q4 and Q5. Nothing experiment-specific here."""
from dataclasses import dataclass, field
from typing import Callable, List

import torch

from src.data import get_batch, make_dataset
from src.model import TinyTransformer
from src.seeding import set_seed
from src.telemetry import UpdateRatioLogger


@dataclass
class RunResult:
    steps: List[int] = field(default_factory=list)
    losses: List[float] = field(default_factory=list)      # eval loss at `steps`
    train_losses: List[float] = field(default_factory=list)
    lrs: List[float] = field(default_factory=list)
    ratio_rows: List[dict] = field(default_factory=list)
    final_loss: float = float("nan")

    def loss_at(self, step: int) -> float:
        """Eval loss at the recorded step nearest to `step` (exact if logged)."""
        i = min(range(len(self.steps)), key=lambda k: abs(self.steps[k] - step))
        return self.losses[i]


@torch.no_grad()
def _eval_loss(model, tokens, batch_size, seq_len, device, n_batches=4):
    model.eval()
    g = torch.Generator().manual_seed(999)  # identical eval batches for every run
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(tokens, batch_size, seq_len, g, device)
        _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / n_batches


def train_run(width: int, peak_lr: float, total_steps: int,
              schedule_fn: Callable[[int, int], float], *,
              seed: int = 1337, batch_size: int = 16, seq_len: int = 128,
              n_layers: int = 2, n_heads: int = 4, log_ratios: bool = False,
              eval_every: int = 10, grad_clip: float = 1.0,
              device: str = "cpu") -> RunResult:
    """Train for total_steps and return the whole trajectory.

    There is deliberately no `stop_at`: schedule_fn(step, total_steps) never looks
    past `step`, so the weights at step k of a truncated run are bit-identical to
    the weights at step k of the full run. Q4 therefore trains the full 300 steps
    and reads loss_at(200) - same answer as stopping at 200, plus the rest of the
    curve for free.
    """
    set_seed(seed)
    tokens, vocab = make_dataset(seq_len=seq_len)
    model = TinyTransformer(vocab, width, n_layers, n_heads, seq_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                            betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    logger = UpdateRatioLogger(model) if log_ratios else None
    batch_gen = torch.Generator().manual_seed(seed)
    res = RunResult()
    horizon = total_steps

    for step in range(1, horizon + 1):
        lr = schedule_fn(step, total_steps)
        for pg in opt.param_groups:
            pg["lr"] = lr
        x, y = get_batch(tokens, batch_size, seq_len, batch_gen, device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if logger is not None:
            logger.snapshot()
        opt.step()
        if logger is not None:
            logger.record(step, lr)

        res.lrs.append(lr)
        res.train_losses.append(loss.item())
        if step % eval_every == 0 or step == horizon or step == 1:
            res.steps.append(step)
            res.losses.append(_eval_loss(model, tokens, batch_size, seq_len, device))

    res.final_loss = res.losses[-1]
    if logger is not None:
        res.ratio_rows = logger.rows
    return res
