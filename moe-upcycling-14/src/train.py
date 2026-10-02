"""One training loop for every run. Writes a human log and a JSONL log per run."""
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from src.seeding import set_seed

EVAL_SEED = 999


@dataclass
class TrainConfig:
    run_name: str
    steps: int
    batch_size: int = 32
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup: int = 100
    weight_decay: float = 0.1
    aux_coef: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 100
    eval_batches: int = 20
    log_every: int = 10
    seed: int = 1337
    step_offset: int = 0


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    progress = min(1.0, (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup))
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, data, cfg: TrainConfig) -> dict:
    was_training = model.training
    model.eval()
    out = {}
    for split in ("train", "val"):
        g = torch.Generator().manual_seed(EVAL_SEED)
        losses = [model(*data.batch(split, cfg.batch_size, model.cfg.seq_len, g))[1].item()
                  for _ in range(cfg.eval_batches)]
        out[split] = sum(losses) / len(losses)
    model.train(was_training)
    return out


def _expert_frac(model):
    fr = [blk.ffn.last_frac.tolist() for blk in model.blocks if hasattr(blk.ffn, "last_frac")]
    return fr or None


def _optimizer(model, cfg):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=cfg.lr, betas=(0.9, 0.95))


def train(model, data, cfg: TrainConfig, log_dir: Path) -> list[dict]:
    set_seed(cfg.seed)
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl = open(log_dir / f"{cfg.run_name}.jsonl", "w", encoding="utf-8")
    human = open(log_dir / f"{cfg.run_name}.log", "w", encoding="utf-8")

    def emit(row: dict):
        rows.append(row)
        jsonl.write(json.dumps(row) + "\n")
        jsonl.flush()
        msg = (f"step {row['step']:5d} (global {row['global_step']:5d}) | lr {row['lr']:.2e} "
               f"| train_ce {row['train_loss']:.4f} | aux {row['aux_loss']:.4f}")
        if "val_loss" in row:
            msg += f" | EVAL train {row['eval_train_loss']:.4f} val {row['val_loss']:.4f}"
        if row.get("expert_frac"):
            msg += " | experts " + " ".join("[" + ",".join(f"{f:.2f}" for f in layer) + "]"
                                            for layer in row["expert_frac"])
        print(msg)
        human.write(msg + "\n")
        human.flush()

    header = (f"# run={cfg.run_name} params={model.num_params():,} "
              f"config={json.dumps(asdict(cfg))} model={json.dumps(asdict(model.cfg))}")
    print(header)
    human.write(header + "\n")

    rows: list[dict] = []
    opt = _optimizer(model, cfg)
    g = torch.Generator().manual_seed(cfg.seed)
    t0 = time.time()
    tokens_per_step = cfg.batch_size * model.cfg.seq_len
    model.train()

    # step 0 = state BEFORE any update (shows the conversion point exactly)
    ev = evaluate(model, data, cfg)
    with torch.no_grad():
        _, ce0 = model(*data.batch("train", cfg.batch_size, model.cfg.seq_len,
                                   torch.Generator().manual_seed(EVAL_SEED)))
    emit({"step": 0, "global_step": cfg.step_offset, "lr": 0.0, "train_loss": ce0.item(),
          "aux_loss": model.aux_loss().item(), "tokens_seen": 0, "elapsed_s": 0.0,
          "eval_train_loss": ev["train"], "val_loss": ev["val"],
          "expert_frac": _expert_frac(model)})

    for step in range(1, cfg.steps + 1):
        lr = lr_at(step - 1, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = data.batch("train", cfg.batch_size, model.cfg.seq_len, g)
        _, ce = model(x, y)
        aux = model.aux_loss()
        loss = ce + cfg.aux_coef * aux
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        is_eval = step % cfg.eval_every == 0 or step == cfg.steps
        if step % cfg.log_every == 0 or is_eval:
            row = {"step": step, "global_step": cfg.step_offset + step, "lr": lr,
                   "train_loss": ce.item(), "aux_loss": aux.item(),
                   "tokens_seen": step * tokens_per_step, "elapsed_s": round(time.time() - t0, 2),
                   "expert_frac": _expert_frac(model)}
            if is_eval:
                ev = evaluate(model, data, cfg)
                row["eval_train_loss"], row["val_loss"] = ev["train"], ev["val"]
            emit(row)

    jsonl.close()
    human.close()
    return rows
