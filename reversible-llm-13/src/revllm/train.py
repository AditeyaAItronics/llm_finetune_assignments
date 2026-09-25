import json
import math
import os
import time
from dataclasses import asdict

import torch

from .config import ModelConfig, TrainConfig
from .data import get_batch, load_split
from .model import GPT
from .reversible import grad_agreement, reconstruction_error, reconstruction_error_rel

EVAL_BATCH = 16  # fixed so the eval sample is identical across runs regardless of training batch size
GRAD_CHECK_BATCH = 4  # fixed val batch (seed 1) used for grad_agreement / recon diagnostics


def lr_at(step: int, total_steps: int, tc: TrainConfig) -> float:
    warmup = min(tc.warmup_steps, max(1, total_steps // 10))
    if step < warmup:
        return tc.lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
    return tc.min_lr + 0.5 * (tc.lr - tc.min_lr) * (1 + math.cos(math.pi * progress))


def amp_dtype() -> torch.dtype:
    # T4 (sm_75) has no native bf16 -> fp16 + GradScaler; A100/L4 (sm_80+) -> bf16.
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


@torch.no_grad()
def evaluate(model, val_data, mc, tc, device, dtype) -> float:
    model.eval()
    gen = torch.Generator().manual_seed(0)
    n_batches = max(1, tc.eval_tokens // (EVAL_BATCH * mc.block_size))
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(val_data, EVAL_BATCH, mc.block_size, device, gen)
        with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / n_batches


def _grad_and_recon_diagnostics(model, val_data, mc, device, dtype, use_amp):
    """grad_agreement + reconstruction_error_rel on a fixed val batch (4 sequences, seed 1),
    inside the same autocast as training. Only meaningful for non-baseline archs."""
    x, y = get_batch(val_data, GRAD_CHECK_BATCH, mc.block_size, device, torch.Generator().manual_seed(1))
    with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
        ga = grad_agreement(model, x, y)
        with torch.no_grad():
            recon_rel = reconstruction_error_rel(model.blocks, model.embed(x))
    return ga["grad_cos"], ga["grad_rel_err"], recon_rel


def _log_eval(model, val_data, mc, tc, device, dtype, use_amp, history, tokens):
    history["val_tokens"].append(tokens)
    history["val_loss"].append(evaluate(model, val_data, mc, tc, device, dtype))
    if mc.arch != "baseline":
        cos, rel_err, recon_rel = _grad_and_recon_diagnostics(model, val_data, mc, device, dtype, use_amp)
        history["grad_cos"].append(cos)
        history["grad_rel_err"].append(rel_err)
        history["recon_rel"].append(recon_rel)


def train(mc: ModelConfig, tc: TrainConfig, max_steps: int | None = None) -> dict:
    torch.manual_seed(tc.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    dtype = amp_dtype()
    train_data, val_data = load_split(tc.data_dir, "train"), load_split(tc.data_dir, "val")

    model = GPT(mc).to(device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tc.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=tc.lr, betas=(0.9, 0.95), fused=device == "cuda",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and dtype == torch.float16)

    tokens_per_step = tc.batch_size * mc.block_size
    total_steps = max_steps or math.ceil(tc.total_tokens / tokens_per_step)
    log_every = max(1, tc.log_every_tokens // tokens_per_step)
    eval_every = max(1, tc.eval_every_tokens // tokens_per_step)
    history = {"tokens": [], "train_loss": [], "val_tokens": [], "val_loss": [],
               "grad_cos": [], "grad_rel_err": [], "recon_rel": []}
    gen = torch.Generator().manual_seed(tc.seed)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    diverged, steps_done = False, 0
    scaler_skipped_steps = 0
    timed_tokens, timed_seconds = 0, 0.0
    t_start = time.time()
    for step in range(total_steps):
        for group in opt.param_groups:
            group["lr"] = lr_at(step, total_steps, tc)
        x, y = get_batch(train_data, tc.batch_size, mc.block_size, device, gen)
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        scaler.step(opt)
        prev_scale = scaler.get_scale()
        scaler.update()
        if scaler.get_scale() < prev_scale:  # GradScaler skipped this optimizer step (inf/nan grads)
            scaler_skipped_steps += 1
        opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        if step >= 10:  # skip warm-up steps (kernel autotune, allocator growth)
            timed_tokens += tokens_per_step
            timed_seconds += time.perf_counter() - t0
        steps_done = step + 1
        tokens = steps_done * tokens_per_step

        if step % log_every == 0 or step == total_steps - 1:
            loss_val = loss.item()
            if not math.isfinite(loss_val):
                diverged = True
                print(f"[{tc.run_name}] loss={loss_val} at step {step}: diverged, stopping")
                break
            history["tokens"].append(tokens)
            history["train_loss"].append(loss_val)
            print(f"[{tc.run_name}] step {step + 1}/{total_steps}  tokens {tokens / 1e6:.1f}M  loss {loss_val:.4f}")
        if step > 0 and step % eval_every == 0 and step != total_steps - 1:
            _log_eval(model, val_data, mc, tc, device, dtype, use_amp, history, tokens)

    if not diverged:
        _log_eval(model, val_data, mc, tc, device, dtype, use_amp, history, steps_done * tokens_per_step)

    recon = None
    if mc.arch != "baseline" and not diverged:
        x, _ = get_batch(val_data, GRAD_CHECK_BATCH, mc.block_size, device, torch.Generator().manual_seed(1))
        with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            recon = reconstruction_error(model.blocks, model.embed(x))

    have_diag = mc.arch != "baseline" and not diverged
    window_tokens = history["tokens"][-5:]
    final_train_loss_window_tokens = (window_tokens[-1] - window_tokens[0]) if len(window_tokens) > 1 \
        else (window_tokens[0] if window_tokens else 0)

    nan = float("nan")
    result = {
        "run_name": tc.run_name, "arch": mc.arch, "h": mc.h, "batch_size": tc.batch_size,
        "params": model.num_params(), "steps": steps_done, "tokens_seen": steps_done * tokens_per_step,
        "diverged": diverged,
        "final_train_loss": nan if diverged else sum(history["train_loss"][-5:]) / len(history["train_loss"][-5:]),
        "final_train_loss_window_tokens": final_train_loss_window_tokens,
        "final_val_loss": nan if diverged else history["val_loss"][-1],
        "tokens_per_sec": timed_tokens / timed_seconds if timed_seconds else None,
        "wall_time_s": time.time() - t_start,
        "peak_mem_allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
        "peak_mem_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if device == "cuda" else None,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "amp_dtype": str(dtype) if use_amp else "float32",
        "scaler_skipped_steps": scaler_skipped_steps,
        "final_loss_scale": scaler.get_scale() if scaler.is_enabled() else None,
        "recon_error": recon,
        "recon_error_rel": history["recon_rel"][-1] if have_diag else None,
        "final_grad_cos": history["grad_cos"][-1] if have_diag else None,
        "final_grad_rel_err": history["grad_rel_err"][-1] if have_diag else None,
        "model_config": asdict(mc), "train_config": asdict(tc), "history": history,
    }
    os.makedirs(tc.out_dir, exist_ok=True)
    with open(os.path.join(tc.out_dir, f"{tc.run_name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result
