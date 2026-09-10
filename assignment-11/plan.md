# Session 11 — Optimizers, Schedules & LR Scaling: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a GitHub repo whose `README.md` answers all five Session 11 questions with reproducible numbers, plots, and a stated judgement call — Adam reproduced by hand, bias correction ablated, update-to-weight ratios logged per layer, cosine vs WSD compared at a truncated horizon, and an LR sweep across three widths extrapolated to width 4096.

**Architecture:** A small shared library (`src/`) holds a hand-written Adam, a width-parameterised tiny decoder-only transformer, LR schedules, and telemetry hooks. Five thin experiment scripts (`experiments/`) each answer exactly one assignment question, write machine-readable results to `results/*.json|csv` and figures to `results/*.png`, and print the exact numbers that get pasted into `README.md`. Unit tests pin the maths (Adam vs PyTorch, schedule shapes, ratio computation) so the experiments can be trusted.

**Tech Stack:** Python 3.10+, PyTorch 2.x (CPU works; a T4/Colab makes Task 9 ~10x faster), NumPy, Matplotlib, pytest.

**Spec:** The assignment brief, reproduced verbatim in the "Assignment Brief" section below. This plan argues from it; executors read both.

---

## Assignment Brief (the spec)

1. **Reproduce Adam by hand.** Take one weight and five gradients, compute `m`, `v`, `m̂`, `v̂` and the resulting step yourself, then check each against PyTorch. They should agree to several decimal places.
2. **Disable bias correction** and plot the first twenty steps both ways. Report the number of steps after which the difference stops mattering.
3. **Log the update-to-weight ratio** for every layer, and identify the step at which warmup stops changing it.
4. **Train the same model twice for 300 steps**, once under cosine and once under WSD, and stop both at step 200. Report both losses and state which model you would keep.
5. **Sweep the learning rate at widths 256, 512 and 1,024**, plot loss against learning rate, and mark the three minima. State the value you would use at width 4,096 and how confident you are in it.

> Tune both sides before accepting a comparison. Almost every optimizer claim that failed to replicate was a well tuned method measured against a badly tuned one.

**Deliverable:** a detailed `README.md` plus supporting code, submitted as a public GitHub link (verified in an incognito window). 1000 points, resubmission allowed, due Sat Sep 12 2026.

---

## Global Constraints

- **Determinism is mandatory.** Every experiment calls `set_seed(1337)` before constructing anything. Any run a grader cannot reproduce is a failed deliverable.
- **Optimizer comparison target for Q1/Q2 is `torch.optim.Adam`, not `AdamW`.** Defaults: `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.0`. Do not enable `amsgrad`, `fused`, or `foreach` when comparing against the hand implementation.
- **Agreement tolerance for Q1:** `abs(mine - torch) < 1e-12` in float64, checked for **each** of `m`, `v`, `m̂`, `v̂` and `w` separately — not just the final weight. The README reports numbers to 12 decimal places.
- **No comparison without a noise floor.** Any claim that A beats B must be accompanied by the within-condition seed spread. A gap smaller than that spread is reported as "no measurable difference", never as a win.
- **Both sides tuned.** Task 8 (cosine vs WSD) must sweep the peak LR for *each* schedule independently and compare best-vs-best. A single-LR comparison is not acceptable and directly violates the brief's warning.
- **Every claim in `README.md` must be traceable** to a file in `results/` produced by a script in `experiments/`. No hand-typed numbers.
- **Model widths for Q5:** exactly 256, 512, 1024. Depth, sequence length, batch size, step count, seed, and data are held constant — width is the only thing that moves.
- **Plots:** matplotlib only, no seaborn, no network-dependent styles. Log-scale x-axis for LR sweeps. Every figure gets axis labels, a title, and a legend.
- **README length target:** 1,500–2,500 words with 4 embedded figures. It is the graded artifact; the code exists to justify it.

---

## What Each Question Is Actually Asking

Executors: read this before writing code. Getting the *intent* wrong is the main failure mode here.

**Q1 (Adam by hand)** is a maths-fidelity check. The trap is `eps` placement. PyTorch computes
`denom = sqrt(v_t)/sqrt(1-β2^t) + eps` and `w -= (lr/(1-β1^t)) * m_t / denom`, which is algebraically `w -= lr * m̂ / (sqrt(v̂) + eps)` — eps is added *outside* the square root of the bias-corrected second moment. If you write `sqrt(v̂ + eps)` you will disagree in the 8th decimal and waste an hour.

**Q2 (bias correction)** is about the early-step transient. With `β1=0.9, β2=0.999` and zero-initialised moments, uncorrected Adam takes a near-zero first step and ramps up; corrected Adam takes a full-size step immediately. The correction factor on the step size is `sqrt(1-β2^t)/(1-β1^t)`. "Stops mattering" needs a *definition you state explicitly*. Report two: (a) the first step after which the two weight trajectories stay within 1% — single digits to low teens; (b) the first step after which the step-size factor stays within 1% of 1.0 — hundreds, because `1-β1^t` saturates by step ~7 but `1-β2^t` with `β2=0.999` needs ~500.

**Q3 (update-to-weight ratio)** is the practical health metric: `||Δw||_RMS / ||w||_RMS` per parameter tensor, per step. Healthy pretraining sits around `1e-3`. During linear warmup the ratio climbs roughly with the LR ramp; once warmup ends it flattens. "The step at which warmup stops changing it" is the knee — detect it programmatically, don't eyeball it.

**Q4 (cosine vs WSD)** is a trick question about *commitment*. Cosine is parameterised by the horizon you declared: a cosine set for 300 steps and killed at 200 is still at ~30–40% of peak LR and has not annealed, so its checkpoint is mid-flight. WSD is in its stable phase at 200, also at high LR — so a naive "stop at 200" comparison is close to a coin flip. The real answer is optionality: WSD lets you *decide at step 200* to spend a short decay and land a good checkpoint; cosine would need the whole schedule re-run. Report the raw step-200 losses (as asked), then report WSD-with-a-decay-from-200 as the honest comparison, and keep the WSD model.

**Q5 (LR sweep across widths)** is µP in disguise. Under standard parameterisation the optimal LR drifts down as width grows; under µP it is width-invariant. Fit `log(lr*) = a + b·log(width)` through the three minima and extrapolate to 4096. With three points and no error bars, confidence is *low* — say so, give an interval, and say what you would do instead (a confirmation point at 2048, or switch to µP so the transfer is free).

---

## File Structure

```
assignment-11/
├── README.md                        # THE deliverable (written in Task 10)
├── plan.md                          # this file
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── seeding.py                   # set_seed(), determinism switches
│   ├── adam_manual.py               # ManualAdam: hand-rolled, bias-correction toggle
│   ├── schedules.py                 # cosine_with_warmup(), wsd(), linear_warmup()
│   ├── model.py                     # TinyTransformer(width=...), width is the only knob
│   ├── data.py                      # deterministic tiny token stream + batcher
│   ├── telemetry.py                 # UpdateRatioLogger: per-layer ||Δw||/||w||
│   ├── train.py                     # one shared training loop
│   └── plotting.py                  # shared figure helpers
├── experiments/
│   ├── __init__.py
│   ├── exp1_adam_by_hand.py         # Q1  -> results/adam_by_hand.{json,md}
│   ├── exp2_bias_correction.py      # Q2  -> results/bias_correction.{csv,json,png}
│   ├── exp3_update_ratio.py         # Q3  -> results/update_ratio.{csv,json,png}
│   ├── exp4_cosine_vs_wsd.py        # Q4  -> results/schedules.{json,png}
│   └── exp5_lr_sweep.py             # Q5  -> results/lr_sweep.{csv,json,png}
├── tests/
│   ├── test_seeding.py
│   ├── test_adam_manual.py
│   ├── test_schedules.py
│   ├── test_telemetry.py
│   └── test_train.py
└── results/                         # committed: figures + json/csv, no checkpoints
```

Rationale: each `src/` module has one responsibility and fits on one screen. Each `experiments/` script is a thin driver — no maths lives there, so the tests over `src/` actually cover the claims in the README.

---

## Task 1: Repo skeleton, seeding, and dependencies

**Files:**
- Create: `requirements.txt`, `.gitignore`, `src/__init__.py`, `src/seeding.py`, `tests/test_seeding.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `set_seed(seed: int = 1337) -> None` — seeds `random`, `numpy`, `torch`, and sets cuDNN determinism flags.

- [ ] **Step 1: Write the failing test**

`tests/test_seeding.py`:
```python
import torch

from src.seeding import set_seed


def test_set_seed_makes_torch_randn_reproducible():
    set_seed(1337)
    a = torch.randn(8)
    set_seed(1337)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    set_seed(1)
    a = torch.randn(8)
    set_seed(2)
    b = torch.randn(8)
    assert not torch.equal(a, b)
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/test_seeding.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src'`.

- [ ] **Step 3: Write the minimal implementation**

`src/__init__.py`: empty file.

`src/seeding.py`:
```python
"""Determinism helpers. Every experiment calls set_seed() first."""
import os
import random

import numpy as np
import torch


def set_seed(seed: int = 1337) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

`requirements.txt`:
```
torch>=2.1
numpy>=1.24
matplotlib>=3.7
pytest>=7.4
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
*.pt
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_seeding.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore src tests
git commit -m "feat(s11): repo skeleton with deterministic seeding"
```

---

## Task 2: Hand-written Adam with a bias-correction toggle

**Files:**
- Create: `src/adam_manual.py`, `tests/test_adam_manual.py`

**Interfaces:**
- Consumes: nothing beyond torch.
- Produces:
  - `adam_step_scalar(w, g, m, v, t, lr, b1, b2, eps, bias_correction=True) -> dict` with keys `t, g, m, v, m_hat, v_hat, update, w_new` — the scalar trace used by Q1's table.
  - `ManualAdam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, bias_correction=True)` with `.step()`, `.zero_grad()`, and `.state[param] -> {"m": Tensor, "v": Tensor}`.

- [ ] **Step 1: Write the failing tests**

`tests/test_adam_manual.py`:
```python
import math

import torch

from src.adam_manual import ManualAdam, adam_step_scalar


def test_scalar_trace_matches_closed_form_first_step():
    out = adam_step_scalar(
        w=0.5, g=0.1, m=0.0, v=0.0, t=1,
        lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, bias_correction=True,
    )
    assert math.isclose(out["m"], 0.1 * 0.1, rel_tol=1e-15)
    assert math.isclose(out["v"], 0.001 * 0.01, rel_tol=1e-15)
    assert math.isclose(out["m_hat"], out["m"] / (1 - 0.9 ** 1), rel_tol=1e-15)
    assert math.isclose(out["v_hat"], out["v"] / (1 - 0.999 ** 1), rel_tol=1e-15)
    expected_update = -1e-3 * out["m_hat"] / (math.sqrt(out["v_hat"]) + 1e-8)
    assert math.isclose(out["update"], expected_update, rel_tol=1e-15)


def test_manual_adam_matches_torch_adam_over_five_steps_float64():
    grads = [0.1, -0.3, 0.05, 0.2, -0.15]

    mine = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    theirs = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

    opt_mine = ManualAdam([mine], lr=1e-3)
    opt_theirs = torch.optim.Adam([theirs], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    for g in grads:
        mine.grad = torch.tensor([g], dtype=torch.float64)
        theirs.grad = torch.tensor([g], dtype=torch.float64)
        opt_mine.step()
        opt_theirs.step()
        assert abs(mine.item() - theirs.item()) < 1e-12


def test_bias_correction_off_takes_a_smaller_first_step():
    w_on = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    w_off = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    on = ManualAdam([w_on], lr=1e-3, bias_correction=True)
    off = ManualAdam([w_off], lr=1e-3, bias_correction=False)
    for w, opt in ((w_on, on), (w_off, off)):
        w.grad = torch.tensor([0.1], dtype=torch.float64)
        opt.step()
    assert abs(w_off.item() - 0.5) < abs(w_on.item() - 0.5)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `python -m pytest tests/test_adam_manual.py -v`
Expected: FAIL, `No module named 'src.adam_manual'`.

- [ ] **Step 3: Write the implementation**

`src/adam_manual.py`:
```python
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
    """Minimal Adam. No weight decay, no amsgrad — matches torch.optim.Adam defaults."""

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
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_adam_manual.py -v`
Expected: 3 passed. If the torch-agreement test fails at ~1e-8, you wrote `sqrt(v_hat + eps)`. Fix the eps placement.

- [ ] **Step 5: Commit**

```bash
git add src/adam_manual.py tests/test_adam_manual.py
git commit -m "feat(s11): hand-written Adam with bias-correction toggle, matches torch to 1e-12"
```

---

## Task 3: Experiment 1 — Adam by hand vs PyTorch (answers Q1)

**Files:**
- Create: `experiments/__init__.py`, `experiments/exp1_adam_by_hand.py`

**Interfaces:**
- Consumes: `adam_step_scalar`, `set_seed`.
- Produces: `results/adam_by_hand.json` (keys `config`, `steps`, `max_abs_diff`, `max_diff_by_quantity`) and `results/adam_by_hand_table.md`, a markdown table pasted straight into the README.

**The brief says "check *each* against PyTorch"** — that means `m`, `v`, `m̂`, `v̂` *and* the step, not just the resulting weight. PyTorch stores the raw moments in `opt.state[p]["exp_avg"]` and `["exp_avg_sq"]` and folds the bias correction into the step size, so `m̂`/`v̂` have to be derived from that state. Comparing only the final weight leaves four of the five requested quantities unchecked.

- [ ] **Step 1: Write the script**

`experiments/__init__.py`: empty file.

`experiments/exp1_adam_by_hand.py`:
```python
"""Q1: one weight, five gradients, m / v / m_hat / v_hat / step by hand vs PyTorch.

Every quantity is checked, not just the final weight. PyTorch keeps the raw
moments in opt.state[p]["exp_avg"] / ["exp_avg_sq"] and folds the bias correction
into the step size, so m_hat / v_hat are derived from that state to compare.
"""
import json
import pathlib

import torch

from src.adam_manual import adam_step_scalar
from src.seeding import set_seed

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
W0, LR, B1, B2, EPS = 0.5, 1e-3, 0.9, 0.999, 1e-8
GRADS = [0.1, -0.3, 0.05, 0.2, -0.15]


def main() -> None:
    set_seed(1337)
    RESULTS.mkdir(exist_ok=True)

    torch_w = torch.tensor([W0], dtype=torch.float64, requires_grad=True)
    torch_opt = torch.optim.Adam([torch_w], lr=LR, betas=(B1, B2), eps=EPS)

    w, m, v, rows = W0, 0.0, 0.0, []
    for t, g in enumerate(GRADS, start=1):
        tr = adam_step_scalar(w, g, m, v, t, LR, B1, B2, EPS, bias_correction=True)
        w, m, v = tr["w_new"], tr["m"], tr["v"]

        torch_w.grad = torch.tensor([g], dtype=torch.float64)
        torch_opt.step()

        # Check EACH quantity against PyTorch, not just the resulting weight.
        st = torch_opt.state[torch_w]
        tr["m_torch"] = st["exp_avg"].item()
        tr["v_torch"] = st["exp_avg_sq"].item()
        tr["m_hat_torch"] = tr["m_torch"] / (1.0 - B1 ** t)
        tr["v_hat_torch"] = tr["v_torch"] / (1.0 - B2 ** t)
        tr["w_mine"] = w
        tr["w_torch"] = torch_w.item()
        tr["m_diff"] = abs(tr["m"] - tr["m_torch"])
        tr["v_diff"] = abs(tr["v"] - tr["v_torch"])
        tr["m_hat_diff"] = abs(tr["m_hat"] - tr["m_hat_torch"])
        tr["v_hat_diff"] = abs(tr["v_hat"] - tr["v_hat_torch"])
        tr["w_diff"] = abs(w - torch_w.item())
        tr["max_diff"] = max(tr["m_diff"], tr["v_diff"], tr["m_hat_diff"],
                             tr["v_hat_diff"], tr["w_diff"])
        rows.append(tr)

    max_abs_diff = max(r["max_diff"] for r in rows)
    assert max_abs_diff < 1e-12, f"disagreement {max_abs_diff:.3e} — check eps placement"

    payload = {
        "config": {"w0": W0, "lr": LR, "betas": [B1, B2], "eps": EPS, "grads": GRADS},
        "steps": rows,
        "max_abs_diff": max_abs_diff,
        "max_diff_by_quantity": {
            q: max(r[f"{q}_diff"] for r in rows)
            for q in ("m", "v", "m_hat", "v_hat", "w")
        },
    }
    (RESULTS / "adam_by_hand.json").write_text(json.dumps(payload, indent=2))

    hdr = ("| t | g | m (mine) | m (torch) | v (mine) | v (torch) | m-hat | v-hat | "
           "update | w (mine) | w (torch) | max diff |")
    sep = "|" + "---|" * 12
    lines = [hdr, sep] + [
        "| {t} | {g:+.2f} | {m:.12f} | {m_torch:.12f} | {v:.12f} | {v_torch:.12f} | "
        "{m_hat:.12f} | {v_hat:.12f} | {update:+.12f} | {w_mine:.12f} | "
        "{w_torch:.12f} | {max_diff:.2e} |".format(**r)
        for r in rows
    ]
    (RESULTS / "adam_by_hand_table.md").write_text("\n".join(lines) + "\n")

    print("\n".join(lines))
    print(f"\nmax |mine - torch| over all quantities = {max_abs_diff:.3e}")
    print(json.dumps(payload["max_diff_by_quantity"], indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python -m experiments.exp1_adam_by_hand`
Expected: a 5-row table printed, and `max |mine - torch| over all quantities` below `1e-12`, with a per-quantity breakdown for `m`, `v`, `m_hat`, `v_hat`, `w`. The in-script assertion fails loudly if not — the most likely cause is `sqrt(v_hat + eps)` instead of `sqrt(v_hat) + eps`, which shows up as a `w` disagreement around `1e-8` while `m` and `v` still match exactly.

- [ ] **Step 3: Sanity-check row 1 against pen and paper**

With `g₁ = 0.1`: `m₁ = 0.01`, `v₁ = 1e-5`, `m̂₁ = 0.1`, `v̂₁ = 0.01`, `update = -1e-3 · 0.1 / (0.1 + 1e-8) ≈ -1.0e-3`. Confirm the JSON agrees. Note for the README: step 1 of Adam is (almost exactly) `-lr · sign(g)` regardless of gradient magnitude — the single most useful intuition in this task.

- [ ] **Step 4: Commit**

```bash
git add experiments results/adam_by_hand*
git commit -m "feat(s11): Q1 experiment — Adam by hand agrees with torch to 1e-12"
```

---

## Task 4: Experiment 2 — bias correction on vs off (answers Q2)

**Files:**
- Create: `src/plotting.py`, `experiments/exp2_bias_correction.py`

**Interfaces:**
- Consumes: `ManualAdam`.
- Produces: `results/bias_correction.csv` (columns `t, w_corrected, w_uncorrected, step_corrected, step_uncorrected, ratio, rel_disp_diff`), `results/bias_correction.png` (2 panels), `results/bias_correction.json` with `t_star_visible` and `t_star_stepsize`.
- Also produces `new_fig(nrows, ncols, figsize)` and `finish(fig, path)` in `src/plotting.py`, used by Tasks 7, 8 and 9.

- [ ] **Step 1: Write the shared plotting helper**

`src/plotting.py`:
```python
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def new_fig(nrows=1, ncols=1, figsize=(10, 4)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, constrained_layout=True)
    return fig, axes


def finish(fig, path):
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")
```

- [ ] **Step 2: Write the experiment**

`experiments/exp2_bias_correction.py`:
```python
"""Q2: bias correction on vs off. First 20 steps plotted both ways.

Two definitions of "stops mattering", both reported:
  t_star_visible  : first t after which |w_on - w_off| / |w_on - w0| stays < 1%
                    for every remaining step in the 20-step window.
  t_star_stepsize : first t after which the per-step scaling factor
                    sqrt(1 - b2**t) / (1 - b1**t) is within 1% of 1.0 forever after.
The second is the honest answer to "when does the correction stop doing anything";
the first is what you actually see in a 20-step plot.
"""
import csv
import json
import math
import pathlib

import torch

from src.adam_manual import ManualAdam
from src.plotting import finish, new_fig
from src.seeding import set_seed

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
W0, LR, B1, B2 = 0.5, 1e-2, 0.9, 0.999
N_PLOT, N_LONG, TOL = 20, 800, 0.01


def main() -> None:
    set_seed(1337)
    RESULTS.mkdir(exist_ok=True)

    w_on = torch.tensor([W0], dtype=torch.float64, requires_grad=True)
    w_off = torch.tensor([W0], dtype=torch.float64, requires_grad=True)
    opt_on = ManualAdam([w_on], lr=LR, betas=(B1, B2), bias_correction=True)
    opt_off = ManualAdam([w_off], lr=LR, betas=(B1, B2), bias_correction=False)

    rows = []
    for t in range(1, N_PLOT + 1):
        g = 0.1  # constant gradient isolates the correction from gradient noise
        prev_on, prev_off = w_on.item(), w_off.item()
        w_on.grad = torch.tensor([g], dtype=torch.float64)
        w_off.grad = torch.tensor([g], dtype=torch.float64)
        opt_on.step()
        opt_off.step()
        disp_on = abs(w_on.item() - W0)
        rel = abs(w_on.item() - w_off.item()) / disp_on if disp_on > 0 else 0.0
        rows.append({
            "t": t,
            "w_corrected": w_on.item(),
            "w_uncorrected": w_off.item(),
            "step_corrected": w_on.item() - prev_on,
            "step_uncorrected": w_off.item() - prev_off,
            "ratio": math.sqrt(1 - B2 ** t) / (1 - B1 ** t),
            "rel_disp_diff": rel,
        })

    over = [r["t"] for r in rows if r["rel_disp_diff"] > TOL]
    t_star_visible = (max(over) + 1) if over else 1

    ratios = [math.sqrt(1 - B2 ** t) / (1 - B1 ** t) for t in range(1, N_LONG + 1)]
    over_long = [t for t, r in enumerate(ratios, start=1) if abs(r - 1.0) > TOL]
    t_star_stepsize = (max(over_long) + 1) if over_long else 1

    with (RESULTS / "bias_correction.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    (RESULTS / "bias_correction.json").write_text(json.dumps({
        "tolerance": TOL,
        "t_star_visible": t_star_visible,
        "t_star_stepsize": t_star_stepsize,
        "final_rel_diff_at_20": rows[-1]["rel_disp_diff"],
    }, indent=2))

    fig, (ax1, ax2) = new_fig(1, 2, figsize=(12, 4.5))
    ts = [r["t"] for r in rows]
    ax1.plot(ts, [r["w_corrected"] for r in rows], "o-", label="bias correction ON")
    ax1.plot(ts, [r["w_uncorrected"] for r in rows], "s--", label="bias correction OFF")
    ax1.axvline(t_star_visible, color="k", ls=":",
                label=f"t* (visible, 1%) = {t_star_visible}")
    ax1.set_xlabel("step t")
    ax1.set_ylabel("weight")
    ax1.set_title("First 20 steps, constant gradient g = 0.1")
    ax1.legend()

    ax2.plot(range(1, N_LONG + 1), ratios)
    ax2.axhline(1.0, color="k", ls="-", lw=0.8)
    ax2.axhline(1.01, color="r", ls=":", lw=0.8)
    ax2.axhline(0.99, color="r", ls=":", lw=0.8, label="+/-1% band")
    ax2.axvline(t_star_stepsize, color="k", ls=":",
                label=f"t* (step size) = {t_star_stepsize}")
    ax2.set_xlabel("step t")
    ax2.set_ylabel(r"$\sqrt{1-\beta_2^t}\,/\,(1-\beta_1^t)$")
    ax2.set_title("Effective step-size correction factor")
    ax2.legend()
    finish(fig, RESULTS / "bias_correction.png")

    print(json.dumps({"t_star_visible": t_star_visible,
                      "t_star_stepsize": t_star_stepsize}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run it**

Run: `python -m experiments.exp2_bias_correction`
Expected: `bias_correction.png` written; `t_star_visible` is single digits to low teens, `t_star_stepsize` is in the hundreds. If `t_star_visible` comes out as 1, your gradient is too small to separate the curves — it should not, with `lr=1e-2`.

- [ ] **Step 4: Draft the answer sentence**

For the README: *"With β=(0.9, 0.999), the two trajectories are within 1% of each other from step `t_star_visible` onward; the step-size correction factor itself is within 1% of unity only from step `t_star_stepsize`. The gap is entirely a β₂ effect — `1-β1^t` is already 0.999 by step 7, while `1-β2^t` needs hundreds of steps."*

- [ ] **Step 5: Commit**

```bash
git add src/plotting.py experiments/exp2_bias_correction.py results/bias_correction*
git commit -m "feat(s11): Q2 experiment — bias-correction ablation over 20 steps"
```

---

## Task 5: Tiny model, data, schedules, and telemetry

**Files:**
- Create: `src/data.py`, `src/model.py`, `src/schedules.py`, `src/telemetry.py`, `tests/test_schedules.py`, `tests/test_telemetry.py`

**Interfaces:**
- Consumes: `set_seed`.
- Produces:
  - `make_dataset(seq_len=128, n_tokens=200_000, seed=1337) -> (tokens: LongTensor, vocab_size: int)`
  - `get_batch(tokens, batch_size, seq_len, generator, device="cpu") -> (x, y)`
  - `TinyTransformer(vocab_size, width, n_layers=2, n_heads=4, seq_len=128)` with `forward(x, y=None) -> (logits, loss)`
  - `cosine_with_warmup(step, peak_lr, warmup_steps, total_steps, min_lr_frac=0.1) -> float`
  - `wsd(step, peak_lr, warmup_steps, decay_start, total_steps, min_lr_frac=0.0) -> float`
  - `linear_warmup(step, warmup_steps) -> float`
  - `UpdateRatioLogger(model)` with `.snapshot()`, `.record(step, lr)`, `.rows -> list[dict(step, layer, lr, w_rms, d_rms, ratio)]`

- [ ] **Step 1: Write the failing schedule tests**

`tests/test_schedules.py`:
```python
import math

from src.schedules import cosine_with_warmup, wsd


def test_cosine_warmup_is_linear_and_hits_peak():
    assert cosine_with_warmup(0, 1e-3, 100, 1000) == 0.0
    assert math.isclose(cosine_with_warmup(50, 1e-3, 100, 1000), 5e-4)
    assert math.isclose(cosine_with_warmup(100, 1e-3, 100, 1000), 1e-3)


def test_cosine_decays_to_min_lr_at_horizon():
    lr = cosine_with_warmup(1000, 1e-3, 100, 1000, min_lr_frac=0.1)
    assert math.isclose(lr, 1e-4, rel_tol=1e-9)


def test_cosine_truncated_at_200_of_300_is_still_well_above_min():
    lr = cosine_with_warmup(200, 1e-3, 30, 300, min_lr_frac=0.1)
    assert 2e-4 < lr < 6e-4


def test_wsd_is_flat_through_the_stable_phase():
    a = wsd(120, 1e-3, warmup_steps=30, decay_start=240, total_steps=300)
    b = wsd(200, 1e-3, warmup_steps=30, decay_start=240, total_steps=300)
    assert math.isclose(a, 1e-3) and math.isclose(b, 1e-3)


def test_wsd_decays_to_zero_at_horizon():
    assert math.isclose(wsd(300, 1e-3, 30, 240, 300, min_lr_frac=0.0), 0.0, abs_tol=1e-12)
```

- [ ] **Step 2: Write the failing telemetry tests**

`tests/test_telemetry.py`:
```python
import torch

from src.telemetry import UpdateRatioLogger


def test_ratio_is_zero_when_nothing_moves():
    model = torch.nn.Linear(4, 4)
    log = UpdateRatioLogger(model)
    log.snapshot()
    log.record(step=1, lr=1e-3)
    assert all(r["ratio"] == 0.0 for r in log.rows)


def test_ratio_matches_hand_computation():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)          # w_rms = 1.0
    log = UpdateRatioLogger(model)
    log.snapshot()
    with torch.no_grad():
        model.weight.add_(0.1)           # d_rms = 0.1
    log.record(step=1, lr=1e-3)
    row = [r for r in log.rows if r["layer"] == "weight"][0]
    assert abs(row["ratio"] - 0.1) < 1e-6


def test_one_row_per_trainable_tensor_per_step():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 3))
    log = UpdateRatioLogger(model)
    n_params = sum(1 for _ in model.parameters())
    for step in range(1, 4):
        log.snapshot()
        log.record(step=step, lr=1e-3)
    assert len(log.rows) == 3 * n_params
```

- [ ] **Step 3: Run both files and confirm they fail**

Run: `python -m pytest tests/test_schedules.py tests/test_telemetry.py -v`
Expected: collection errors — `No module named 'src.schedules'`, `No module named 'src.telemetry'`.

- [ ] **Step 4: Write `src/schedules.py`**

```python
"""LR schedules. Every one returns a plain float given an integer step."""
import math


def linear_warmup(step: int, warmup_steps: int) -> float:
    if warmup_steps <= 0:
        return 1.0
    return min(1.0, step / warmup_steps)


def cosine_with_warmup(step, peak_lr, warmup_steps, total_steps, min_lr_frac=0.1):
    """Linear warmup then cosine decay to min_lr_frac * peak_lr at total_steps.

    Key property for Q4: this curve is defined by total_steps. Truncating the run
    early does NOT give you the annealed endpoint — you land mid-decay.
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
```

- [ ] **Step 5: Write `src/telemetry.py`**

```python
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
```

- [ ] **Step 6: Write `src/data.py`**

```python
"""A deterministic, dependency-free token stream.

We do NOT download a corpus: graders must be able to run this offline and get
identical numbers. The stream is a synthetic mixture of repeated motifs plus
noise, which gives a loss curve with real structure (learnable, not trivial).
"""
import torch

VOCAB_SIZE = 256


def make_dataset(seq_len: int = 128, n_tokens: int = 200_000, seed: int = 1337):
    g = torch.Generator().manual_seed(seed)
    motifs = [torch.randint(0, VOCAB_SIZE, (k,), generator=g) for k in (3, 5, 8, 13)]
    out, total = [], 0
    while total < n_tokens:
        if torch.rand(1, generator=g).item() < 0.7:
            m = motifs[torch.randint(0, len(motifs), (1,), generator=g).item()]
            out.append(m)
            total += m.numel()
        else:
            out.append(torch.randint(0, VOCAB_SIZE, (4,), generator=g))
            total += 4
    tokens = torch.cat(out)[:n_tokens].long()
    assert tokens.numel() > seq_len + 1
    return tokens, VOCAB_SIZE


def get_batch(tokens: torch.Tensor, batch_size: int, seq_len: int,
              generator: torch.Generator, device: str = "cpu"):
    hi = tokens.numel() - seq_len - 1
    idx = torch.randint(0, hi, (batch_size,), generator=generator)
    x = torch.stack([tokens[i:i + seq_len] for i in idx])
    y = torch.stack([tokens[i + 1:i + seq_len + 1] for i in idx])
    return x.to(device), y.to(device)
```

- [ ] **Step 7: Write `src/model.py`**

```python
"""A tiny decoder-only transformer whose only knob is `width`.

Depth, heads, seq_len and vocab are held fixed across the width sweep so that
Q5's LR-vs-width curve isolates width.

KNOWN CONFOUND, to be named in the README: with n_heads fixed at 4, head_dim
scales with width (64 -> 128 -> 256), so the attention logit scale 1/sqrt(head_dim)
moves along with width. Fixing head_dim and scaling n_heads instead is the other
defensible convention; muP fixes head_dim. Whichever you pick, one variable rides
along with width, so state which one you chose and that lr* absorbs it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, width: int, n_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, n_heads, batch_first=True, bias=False)
        self.ln2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width, bias=False),
            nn.GELU(),
            nn.Linear(4 * width, width, bias=False),
        )

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, width: int, n_layers: int = 2,
                 n_heads: int = 4, seq_len: int = 128):
        super().__init__()
        self.seq_len = seq_len
        self.tok = nn.Embedding(vocab_size, width)
        self.pos = nn.Embedding(seq_len, width)
        self.blocks = nn.ModuleList([Block(width, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(width)
        self.head = nn.Linear(width, vocab_size, bias=False)
        self.apply(self._init)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask, persistent=False)

    @staticmethod
    def _init(m):
        # Standard parameterisation: std = 1/sqrt(fan_in). Q5 measures how the
        # optimal LR drifts with width UNDER THIS CHOICE. (muP would remove the drift.)
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(m.in_features))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, x, y=None):
        b, t = x.shape
        pos = torch.arange(t, device=x.device)
        h = self.tok(x) + self.pos(pos)[None]
        m = self.mask[:t, :t]
        for blk in self.blocks:
            h = blk(h, m)
        logits = self.head(self.ln_f(h))
        if y is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        return logits, loss
```

- [ ] **Step 8: Run all tests and confirm they pass**

Run: `python -m pytest tests/ -v`
Expected: all green.

- [ ] **Step 9: Smoke-test the model shapes and initial loss**

Run:
```bash
python -c "import torch; from src.seeding import set_seed; from src.data import make_dataset, get_batch; from src.model import TinyTransformer; set_seed(1337); tok, V = make_dataset(); g = torch.Generator().manual_seed(0); x, y = get_batch(tok, 8, 128, g); m = TinyTransformer(V, width=256); _, loss = m(x, y); print('params', sum(p.numel() for p in m.parameters()), 'loss', loss.item())"
```
Expected: initial loss near `ln(256) ≈ 5.545`. If it is far off, initialisation is wrong — fix it before proceeding, because every later number depends on it.

- [ ] **Step 10: Commit**

```bash
git add src tests
git commit -m "feat(s11): tiny transformer, synthetic data, LR schedules, update-ratio telemetry"
```

---

## Task 6: Shared training loop

**Files:**
- Create: `src/train.py`, `tests/test_train.py`

**Interfaces:**
- Consumes: `make_dataset`, `get_batch`, `TinyTransformer`, `UpdateRatioLogger`, `set_seed`.
- Produces: `train_run(width, peak_lr, total_steps, schedule_fn, *, seed=1337, batch_size=16, seq_len=128, n_layers=2, n_heads=4, log_ratios=False, eval_every=10, grad_clip=1.0, device="cpu") -> RunResult`, where `RunResult` is a dataclass with `steps`, `losses`, `train_losses`, `lrs`, `ratio_rows`, `final_loss`, and `loss_at(step) -> float`. `schedule_fn` has signature `(step: int, total_steps: int) -> float`.

There is no `stop_at` parameter by design — see the docstring in Step 3. Truncating a run is the same as reading an earlier point of the full run, because the schedule never looks past the current step.

- [ ] **Step 1: Write the failing test**

`tests/test_train.py`:
```python
from src.schedules import cosine_with_warmup
from src.train import train_run


def _sched(step, total):
    return cosine_with_warmup(step, 1e-3, warmup_steps=5, total_steps=total)


def test_short_run_reduces_loss():
    r = train_run(width=64, peak_lr=1e-3, total_steps=40,
                  schedule_fn=_sched, batch_size=8, seq_len=32)
    assert r.losses[0] > r.losses[-1]


def test_run_is_deterministic():
    kw = dict(width=64, peak_lr=1e-3, total_steps=20,
              schedule_fn=_sched, batch_size=8, seq_len=32)
    a = train_run(**kw)
    b = train_run(**kw)
    assert a.final_loss == b.final_loss


def test_ratio_rows_populated_when_requested():
    r = train_run(width=64, peak_lr=1e-3, total_steps=10, schedule_fn=_sched,
                  batch_size=8, seq_len=32, log_ratios=True)
    assert len(r.ratio_rows) > 0
    assert {"step", "layer", "ratio", "lr"} <= set(r.ratio_rows[0])
```

- [ ] **Step 2: Run it and confirm it fails**

Run: `python -m pytest tests/test_train.py -v`
Expected: FAIL, `No module named 'src.train'`.

- [ ] **Step 3: Write the implementation**

`src/train.py`:
```python
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
    and reads loss_at(200) — same answer as stopping at 200, plus the rest of the
    curve for free. Say this in the README so it does not read as ignoring the
    "stop both at step 200" instruction.
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
```

**Note on the optimizer split (put this in the README so it does not read as an inconsistency):** Q1/Q2 compare against `torch.optim.Adam` with textbook defaults on a single scalar, because the question is about Adam's arithmetic. Q3–Q5 train a real model with `AdamW(betas=(0.9, 0.95), weight_decay=0.1)` because that is the actual LLM-pretraining recipe.

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `python -m pytest tests/test_train.py -v`
Expected: 3 passed (~30s on CPU).

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train.py
git commit -m "feat(s11): shared deterministic training loop with optional ratio logging"
```

---

## Task 7: Experiment 3 — update-to-weight ratio and the warmup knee (answers Q3)

**Files:**
- Create: `experiments/exp3_update_ratio.py`

**Interfaces:**
- Consumes: `train_run(..., log_ratios=True)`, `wsd`, `new_fig`, `finish`.
- Produces: `results/update_ratio.csv` (all rows), `results/update_ratio.png`, `results/update_ratio.json` with `warmup_steps`, `knee_step_per_layer`, `median_knee_step`, `median_ratio_after_warmup`.

- [ ] **Step 1: Write the experiment**

`experiments/exp3_update_ratio.py`:
```python
"""Q3: log ||dw||/||w|| for every layer; find the step where warmup stops moving it.

Detection rule (stated so it is reproducible, not eyeballed): smooth each layer's
ratio with a centred 5-step moving average, then find the first step k such that
for every j >= k in the stable phase, |s[j] - s[k]| / s[k] < 15%. That k is the
knee: the point after which the ratio is no longer being dragged by the LR ramp.
"""
import csv
import json
import pathlib
from collections import defaultdict

from src.plotting import finish, new_fig
from src.schedules import wsd
from src.train import train_run

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
TOTAL, WARMUP, DECAY_START, PEAK_LR, WIDTH = 300, 60, 240, 3e-3, 256


def sched(step, total):
    return wsd(step, PEAK_LR, warmup_steps=WARMUP,
               decay_start=DECAY_START, total_steps=total)


def smooth(xs, k=5):
    out = []
    for i in range(len(xs)):
        lo, hi = max(0, i - k // 2), min(len(xs), i + k // 2 + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def find_knee(steps, ratios, stable_end, tol=0.15):
    s = smooth(ratios)
    idx = [i for i, st in enumerate(steps) if st <= stable_end]
    for i in idx:
        if s[i] <= 0:
            continue
        if all(abs(s[j] - s[i]) / s[i] < tol for j in idx if j >= i):
            return steps[i]
    return steps[idx[-1]]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    res = train_run(width=WIDTH, peak_lr=PEAK_LR, total_steps=TOTAL,
                    schedule_fn=sched, log_ratios=True, eval_every=25)

    with (RESULTS / "update_ratio.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(res.ratio_rows[0]))
        wr.writeheader()
        wr.writerows(res.ratio_rows)

    by_layer = defaultdict(lambda: ([], []))
    for r in res.ratio_rows:
        by_layer[r["layer"]][0].append(r["step"])
        by_layer[r["layer"]][1].append(r["ratio"])

    knees = {n: find_knee(st, ra, DECAY_START) for n, (st, ra) in by_layer.items()}
    med_knee = sorted(knees.values())[len(knees) // 2]
    after = [r["ratio"] for r in res.ratio_rows if WARMUP <= r["step"] < DECAY_START]
    med_ratio = sorted(after)[len(after) // 2]

    (RESULTS / "update_ratio.json").write_text(json.dumps({
        "width": WIDTH, "peak_lr": PEAK_LR, "warmup_steps": WARMUP,
        "decay_start": DECAY_START, "total_steps": TOTAL,
        "knee_step_per_layer": knees,
        "median_knee_step": med_knee,
        "median_ratio_after_warmup": med_ratio,
    }, indent=2))

    fig, ax = new_fig(figsize=(11, 5.5))
    for name, (st, ra) in sorted(by_layer.items()):
        ax.plot(st, smooth(ra), lw=1.2, label=name)
    ax.axvline(WARMUP, color="k", ls="--", label=f"warmup ends (step {WARMUP})")
    ax.axvline(med_knee, color="r", ls=":", label=f"median knee (step {med_knee})")
    ax.axvline(DECAY_START, color="grey", ls="-.", label="decay starts")
    ax.axhline(1e-3, color="green", ls=":", lw=1, label="1e-3 rule of thumb")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\|\Delta w\|_{RMS} / \|w\|_{RMS}$")
    ax.set_title(f"Update-to-weight ratio per layer (width {WIDTH}, WSD, peak lr {PEAK_LR})")
    ax.legend(fontsize=7, ncol=2)
    finish(fig, RESULTS / "update_ratio.png")

    print(json.dumps({"median_knee_step": med_knee,
                      "median_ratio_after_warmup": med_ratio}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python -m experiments.exp3_update_ratio`
Expected: ~2–5 min on CPU. `median_knee_step` should land near `WARMUP = 60` (roughly 40–90 is defensible). `median_ratio_after_warmup` should be within about an order of magnitude of `1e-3`.

- [ ] **Step 3: Inspect the figure and note the per-layer story**

Open `results/update_ratio.png`. Record for the README: which layers sit *above* 1e-3 (usually LayerNorm gains and the embedding, because their weight RMS is small), which sit below, and whether any layer's knee is much later than the others. If every curve is flat from step 1, warmup is not being applied — print `res.lrs[:70]` and check.

- [ ] **Step 4: Commit**

```bash
git add experiments/exp3_update_ratio.py results/update_ratio*
git commit -m "feat(s11): Q3 experiment — per-layer update/weight ratio and warmup knee detection"
```

---

## Task 8: Experiment 4 — cosine vs WSD, both tuned, truncated at 200 (answers Q4)

**Files:**
- Create: `experiments/exp4_cosine_vs_wsd.py`

**Interfaces:**
- Consumes: `train_run`, `cosine_with_warmup`, `wsd`.
- Produces: `results/schedules.json` with `tuning` (peak-LR sweep per schedule), `best_peak_lr`, `loss_at_200`, `loss_at_300`, `lr_at_200`, `wsd_decay_from_200`, `seed_spread`, `gap_at_200`, `gap_exceeds_seed_noise`; and `results/schedules.png` (2 panels: LR curves; eval-loss curves with step 200 marked).

- [ ] **Step 1: Write the experiment**

`experiments/exp4_cosine_vs_wsd.py`:
```python
"""Q4: same model, 300 steps, cosine vs WSD, both stopped at step 200.

The brief's warning applies here: tune BOTH schedules before comparing. We sweep
the peak LR for each schedule over the same grid, pick each schedule's own best,
and only then compare. A third run answers what the assignment is really poking
at: WSD lets you decide at step 200 to spend a short decay; cosine cannot.
"""
import json
import pathlib

from src.plotting import finish, new_fig
from src.schedules import cosine_with_warmup, wsd
from src.train import train_run

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
WIDTH, TOTAL, WARMUP, STOP = 256, 300, 30, 200
LR_GRID = [3e-4, 1e-3, 3e-3, 6e-3, 1e-2]
SEEDS = [1337, 2024, 7]  # within-schedule spread, to size the between-schedule gap


def cosine_sched(peak):
    return lambda step, total: cosine_with_warmup(
        step, peak, warmup_steps=WARMUP, total_steps=total, min_lr_frac=0.1)


def wsd_sched(peak, decay_start=240, total_override=TOTAL):
    return lambda step, total: wsd(
        step, peak, warmup_steps=WARMUP, decay_start=decay_start,
        total_steps=total_override, min_lr_frac=0.0)


def tune(make_sched):
    """Sweep peak LR over the full 300-step horizon; return {lr: final_loss}."""
    out = {}
    for lr in LR_GRID:
        r = train_run(width=WIDTH, peak_lr=lr, total_steps=TOTAL,
                      schedule_fn=make_sched(lr), eval_every=25)
        out[lr] = r.final_loss
        print(f"  peak_lr={lr:g}  final_loss={r.final_loss:.4f}")
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    print("tuning cosine:")
    cos_tune = tune(cosine_sched)
    print("tuning wsd:")
    wsd_tune = tune(lambda lr: wsd_sched(lr))

    best_cos = min(cos_tune, key=cos_tune.get)
    best_wsd = min(wsd_tune, key=wsd_tune.get)
    print(f"best cosine peak_lr={best_cos:g}   best wsd peak_lr={best_wsd:g}")

    cos = train_run(width=WIDTH, peak_lr=best_cos, total_steps=TOTAL,
                    schedule_fn=cosine_sched(best_cos), eval_every=10)
    wsd_run = train_run(width=WIDTH, peak_lr=best_wsd, total_steps=TOTAL,
                        schedule_fn=wsd_sched(best_wsd), eval_every=10)
    # The honest comparison: WSD that decides at 200 to decay over 200 -> 240.
    wsd_early = train_run(width=WIDTH, peak_lr=best_wsd, total_steps=240,
                          schedule_fn=wsd_sched(best_wsd, decay_start=200,
                                                total_override=240),
                          eval_every=10)

    # Seed spread at each schedule's own best LR. Without this the step-200
    # comparison is unfalsifiable: a gap smaller than the within-schedule spread
    # is not a result, and saying so is the correct answer to this question.
    print("measuring seed spread:")
    spread = {}
    for name, make, best in (("cosine", cosine_sched, best_cos),
                             ("wsd", wsd_sched, best_wsd)):
        vals = [train_run(width=WIDTH, peak_lr=best, total_steps=TOTAL,
                          schedule_fn=make(best), seed=s,
                          eval_every=10).loss_at(STOP) for s in SEEDS]
        spread[name] = {"seeds": SEEDS, "losses": vals,
                        "range": max(vals) - min(vals),
                        "mean": sum(vals) / len(vals)}
        print(f"  {name}: {['%.4f' % v for v in vals]} range={spread[name]['range']:.4f}")

    gap = abs(cos.loss_at(STOP) - wsd_run.loss_at(STOP))
    worst_spread = max(spread[k]["range"] for k in spread)

    payload = {
        "config": {"width": WIDTH, "total_steps": TOTAL, "warmup": WARMUP,
                   "report_at_step": STOP, "lr_grid": LR_GRID, "seeds": SEEDS},
        "tuning": {"cosine": cos_tune, "wsd": wsd_tune},
        "best_peak_lr": {"cosine": best_cos, "wsd": best_wsd},
        "loss_at_200": {"cosine": cos.loss_at(STOP), "wsd": wsd_run.loss_at(STOP)},
        "loss_at_300": {"cosine": cos.final_loss, "wsd": wsd_run.final_loss},
        "lr_at_200": {"cosine": cos.lrs[STOP - 1], "wsd": wsd_run.lrs[STOP - 1]},
        "wsd_decay_from_200": {"final_step": 240, "loss": wsd_early.final_loss},
        "seed_spread": spread,
        "gap_at_200": gap,
        "gap_exceeds_seed_noise": gap > worst_spread,
    }
    (RESULTS / "schedules.json").write_text(json.dumps(payload, indent=2))

    fig, (ax1, ax2) = new_fig(1, 2, figsize=(12, 4.5))
    ax1.plot(range(1, len(cos.lrs) + 1), cos.lrs, label=f"cosine (peak {best_cos:g})")
    ax1.plot(range(1, len(wsd_run.lrs) + 1), wsd_run.lrs, label=f"WSD (peak {best_wsd:g})")
    ax1.plot(range(1, len(wsd_early.lrs) + 1), wsd_early.lrs, ls="--",
             label="WSD, decay from 200")
    ax1.axvline(STOP, color="k", ls=":", label="stop at 200")
    ax1.set_xlabel("step")
    ax1.set_ylabel("learning rate")
    ax1.set_title("Schedules")
    ax1.legend(fontsize=8)

    ax2.plot(cos.steps, cos.losses, label="cosine")
    ax2.plot(wsd_run.steps, wsd_run.losses, label="WSD")
    ax2.plot(wsd_early.steps, wsd_early.losses, ls="--", label="WSD, decay from 200")
    ax2.axvline(STOP, color="k", ls=":", label="stop at 200")
    ax2.set_xlabel("step")
    ax2.set_ylabel("eval loss")
    ax2.set_title("Loss, best peak LR for each schedule")
    ax2.legend(fontsize=8)
    finish(fig, RESULTS / "schedules.png")

    print(json.dumps({"loss_at_200": payload["loss_at_200"],
                      "gap_at_200": gap,
                      "worst_seed_range": worst_spread,
                      "gap_exceeds_seed_noise": payload["gap_exceeds_seed_noise"]},
                     indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python -m experiments.exp4_cosine_vs_wsd`
Expected: 19 runs of ~300 steps (10 tuning + 3 headline + 6 seed-spread). ~30–55 min on CPU, ~6 min on a T4. If CPU time is a problem, cut `LR_GRID` to 4 values and `seq_len` to 64 — but change it *once, before running*, and record the change in the README config table. Do not drop the seed-spread runs to save time; they are what makes the answer falsifiable.

- [ ] **Step 3: Read `gap_exceeds_seed_noise` before writing a word of the answer**

If it is `false`, the two schedules did not measurably differ at step 200 on this setup, and the README must say exactly that — with both numbers and the seed range next to them. That is a correct and well-supported answer, and a better one than declaring a winner the data does not support. The optionality argument in Step 4 stands either way, because it is an argument about what each schedule *lets you do next*, not about which step-200 loss is lower.

- [ ] **Step 4: Write down the answer**

For the README: report `loss_at_200` for both, report `lr_at_200` for both (this is the explanatory variable), report `wsd_decay_from_200`, and state the keep decision in one sentence with its reason. Expected shape of the argument: *cosine at step 200 of a 300-step schedule is still at ~30–40% of peak LR and has not annealed, so its checkpoint is mid-flight; WSD at 200 is at full peak LR, so its raw step-200 loss is no better. The difference is optionality — WSD reaches a properly annealed checkpoint 40 steps after you decide to stop, and cosine would need the whole schedule re-run. Keep the WSD model.* Do not paste that sentence if your numbers say otherwise — report what you measured.

- [ ] **Step 5: Commit**

```bash
git add experiments/exp4_cosine_vs_wsd.py results/schedules*
git commit -m "feat(s11): Q4 experiment — cosine vs WSD, both peak-LR tuned, truncated at 200"
```

---

## Task 9: Experiment 5 — LR sweep across widths 256/512/1024 (answers Q5)

**Files:**
- Create: `experiments/exp5_lr_sweep.py`

**Interfaces:**
- Consumes: `train_run`, `wsd`.
- Produces: `results/lr_sweep.csv` (`width, lr, final_loss`), `results/lr_sweep.png` (2 panels: loss vs LR with three minima marked; `lr*` vs width with the fitted line extrapolated to 4096), `results/lr_sweep.json` with `minima` (each carrying `lr_grid_argmin`, `lr_star_refined`, `loss_at_min`, `n_diverged`, `min_is_interior`), `fit`, `lr_pred_4096`, `lr_pred_4096_interval`, `factor_per_width_doubling`.

- [ ] **Step 1: Write the experiment**

`experiments/exp5_lr_sweep.py`:
```python
"""Q5: LR sweep at widths 256/512/1024, minima marked, extrapolated to 4096.

Everything except width is held fixed. We fit log10(lr*) = a + b*log2(width)
through the three minima and extrapolate. With three points and no repeats the
fit is weakly determined — the JSON records an interval, and the README says so.
"""
import csv
import json
import math
import pathlib

from src.plotting import finish, new_fig
from src.schedules import wsd
from src.train import train_run

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
WIDTHS = [256, 512, 1024]
LR_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2]
STEPS, WARMUP = 200, 20


def sched(peak):
    return lambda step, total: wsd(step, peak, warmup_steps=WARMUP,
                                   decay_start=int(0.8 * total),
                                   total_steps=total, min_lr_frac=0.0)


def parabola_min(xs, ys):
    """Refine the minimum by fitting a parabola in log10(lr) to the best 3 points.

    Callers MUST pass only finite losses. Diverged runs are excluded upstream: a
    single NaN-sentinel neighbour dominates the fit and returns a plausible-looking
    but meaningless lr*, which then propagates into the width-4096 extrapolation.
    """
    assert all(math.isfinite(y) for y in ys), "parabola_min needs finite losses only"
    i = min(range(len(ys)), key=lambda k: ys[k])
    lo, hi = max(0, i - 1), min(len(xs), i + 2)
    if hi - lo < 3:
        return xs[i]
    lx = [math.log10(x) for x in xs[lo:hi]]
    ly = ys[lo:hi]
    (x0, x1, x2), (y0, y1, y2) = lx, ly
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if d == 0:
        return xs[i]
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / d
    if a <= 0:
        return xs[i]
    return 10 ** (-b / (2 * a))


def linfit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return a, b, r2


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    rows, curves = [], {}

    for w in WIDTHS:
        losses = []
        for lr in LR_GRID:
            r = train_run(width=w, peak_lr=lr, total_steps=STEPS,
                          schedule_fn=sched(lr), eval_every=50)
            loss = r.final_loss if math.isfinite(r.final_loss) else float("nan")
            losses.append(loss)
            rows.append({"width": w, "lr": lr, "final_loss": loss})
            print(f"width={w:4d} lr={lr:g} loss={loss:.4f}")
        curves[w] = losses

    with (RESULTS / "lr_sweep.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["width", "lr", "final_loss"])
        wr.writeheader()
        wr.writerows(rows)

    minima = {}
    for w in WIDTHS:
        # Diverged runs stay in the CSV and on the plot (they mark the divergence
        # boundary and are worth reporting) but they must never enter the fit.
        finite = [(lr, y) for lr, y in zip(LR_GRID, curves[w]) if math.isfinite(y)]
        assert len(finite) >= 3, f"width {w}: fewer than 3 finite points, widen LR_GRID"
        lrs_f = [lr for lr, _ in finite]
        ys_f = [y for _, y in finite]
        i_best = min(range(len(ys_f)), key=lambda k: ys_f[k])
        minima[w] = {
            "lr_grid_argmin": lrs_f[i_best],
            "lr_star_refined": parabola_min(lrs_f, ys_f),
            "loss_at_min": ys_f[i_best],
            "n_diverged": len(LR_GRID) - len(finite),
            "min_is_interior": 0 < i_best < len(ys_f) - 1,
        }

    xs = [math.log2(w) for w in WIDTHS]
    ys = [math.log10(minima[w]["lr_star_refined"]) for w in WIDTHS]
    a, b, r2 = linfit(xs, ys)
    pred = 10 ** (a + b * math.log2(4096))
    # Crude interval: refit dropping each point in turn, take the spread.
    loo = []
    for k in range(3):
        xs2 = [x for j, x in enumerate(xs) if j != k]
        ys2 = [y for j, y in enumerate(ys) if j != k]
        a2, b2, _ = linfit(xs2, ys2)
        loo.append(10 ** (a2 + b2 * math.log2(4096)))

    payload = {
        "config": {"widths": WIDTHS, "lr_grid": LR_GRID, "steps": STEPS,
                   "warmup": WARMUP,
                   "parameterisation": "standard (std = 1/sqrt(fan_in))"},
        "minima": minima,
        "fit": {"form": "log10(lr*) = a + b*log2(width)", "a": a, "b": b, "r2": r2},
        "lr_pred_4096": pred,
        "lr_pred_4096_interval": [min(loo), max(loo)],
        "factor_per_width_doubling": 10 ** b,
    }
    (RESULTS / "lr_sweep.json").write_text(json.dumps(payload, indent=2))

    fig, (ax1, ax2) = new_fig(1, 2, figsize=(12.5, 4.8))
    for w in WIDTHS:
        ax1.plot(LR_GRID, curves[w], "o-", label=f"width {w}")
        m = minima[w]
        ax1.axvline(m["lr_star_refined"], ls=":", lw=1)
        ax1.annotate(f"lr*={m['lr_star_refined']:.2e}",
                     (m["lr_star_refined"], m["loss_at_min"]),
                     textcoords="offset points", xytext=(5, 8), fontsize=8)
    ax1.set_xscale("log")
    ax1.set_xlabel("peak learning rate")
    ax1.set_ylabel(f"eval loss @ {STEPS} steps")
    ax1.set_title("LR sweep by width (three minima marked)")
    ax1.legend()

    ax2.plot(WIDTHS, [minima[w]["lr_star_refined"] for w in WIDTHS], "o", ms=9,
             label="measured lr*")
    grid = [256, 512, 1024, 2048, 4096]
    ax2.plot(grid, [10 ** (a + b * math.log2(w)) for w in grid], "--",
             label=f"fit: slope {b:.2f} per log2(width)")
    ax2.plot([4096], [pred], "*", ms=16, label=f"predicted @4096 = {pred:.2e}")
    ax2.vlines(4096, min(loo), max(loo), lw=6, alpha=0.3, label="leave-one-out range")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("width")
    ax2.set_ylabel("optimal peak LR")
    ax2.set_title("Optimal LR vs width, extrapolated")
    ax2.legend(fontsize=8)
    finish(fig, RESULTS / "lr_sweep.png")

    print(json.dumps({"lr_pred_4096": pred,
                      "interval": [min(loo), max(loo)], "r2": r2}, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it**

Run: `python -m experiments.exp5_lr_sweep`
Expected: 21 runs of 200 steps; width 1024 dominates the cost. ~45–90 min on CPU, ~6 min on a T4. Diverged runs (`nan`) at the top of the grid are *expected and informative* — keep them in the CSV, and say in the README that the right edge of each curve is the divergence boundary.

- [ ] **Step 3: Validate the sweep is well-formed**

Three checks, all blocking:
1. Each width's curve is U-shaped with the minimum **strictly inside** the grid — read `min_is_interior` in `lr_sweep.json`, which must be `true` for all three widths. If a minimum sits at an endpoint, extend `LR_GRID` on that side and re-run; an endpoint minimum makes the extrapolation meaningless. Also check `n_diverged`: if it is 0 for a width you never found that width's instability boundary, and if it is more than 2 or 3 your grid is mostly wasted above the cliff.
2. The fitted slope `b` is **negative** (optimal LR falls as width grows under standard parameterisation). A positive slope means the init is probably not fan-in scaled — check `TinyTransformer._init`.
3. `r2` is reported honestly. Three points will often give a high `r2` that means nothing; do not present it as evidence of a good fit.

- [ ] **Step 4: Commit**

```bash
git add experiments/exp5_lr_sweep.py results/lr_sweep*
git commit -m "feat(s11): Q5 experiment — LR sweep at widths 256/512/1024 with width extrapolation"
```

---

## Task 10: Write `README.md` (the graded deliverable)

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: every file in `results/`.
- Produces: the submission artifact.

- [ ] **Step 1: Build the skeleton with this exact section order**

```markdown
# Session 11 — Adam, Bias Correction, Warmup, Schedules and LR Scaling

## TL;DR — the five answers
| # | Question | Answer |
|---|---|---|
| 1 | Adam by hand vs PyTorch | agree to <X> decimal places (max abs diff <D>) |
| 2 | When bias correction stops mattering | step <A> (visible, 1%) / step <B> (step-size factor) |
| 3 | When warmup stops moving the update ratio | step <K>, vs warmup end at step <W> |
| 4 | Cosine vs WSD stopped at 200 | <L_cos> vs <L_wsd>; I keep <which> because <why> |
| 5 | LR at width 4096 | <LR>, low confidence, interval [<lo>, <hi>] |

## How to reproduce
## Setup: model, data, optimizer, and what is held fixed
## Q1 — Adam by hand
## Q2 — Bias correction on vs off
## Q3 — Update-to-weight ratio per layer
## Q4 — Cosine vs WSD at a truncated horizon
## Q5 — LR sweep across widths and the jump to 4096
## What I would do differently with more compute
## Repo layout
```

- [ ] **Step 2: Fill Q1**

Paste `results/adam_by_hand_table.md` verbatim. Above it, write the update rule in LaTeX. Below it: the `max_diff_by_quantity` breakdown from `adam_by_hand.json` (this is what shows you checked *each* quantity, not just the weight), the eps-placement note (`sqrt(v̂) + eps`, not `sqrt(v̂ + eps)`), a sentence on how PyTorch stores raw moments and folds the correction into the step size so `m̂`/`v̂` had to be derived from `exp_avg`/`exp_avg_sq`, and the "step 1 ≈ `-lr · sign(g)`" observation.

- [ ] **Step 3: Fill Q2**

Embed `results/bias_correction.png`. State both `t_star_visible` and `t_star_stepsize`, **and state the definition used for each** — the assignment asks for "the number of steps after which the difference stops mattering", and an unqualified number is not an answer. Explain that `1-β1^t` saturates in ~7 steps while `1-β2^t` takes hundreds, so the answer depends entirely on which you measure.

- [ ] **Step 4: Fill Q3**

Embed `results/update_ratio.png`. Give the per-layer knee table from `update_ratio.json`, the median knee, and how it compares to `warmup_steps = 60`. State the detection rule in one sentence. Add the practical reading: which layers ride above 1e-3, which below, and what you would change if a layer sat at 1e-1 (lower LR, or check init) or 1e-5 (layer is dead).

- [ ] **Step 5: Fill Q4**

Embed `results/schedules.png`. Put the tuning table (peak LR × schedule → final loss) **first**, to show both sides were tuned — this directly answers the brief's warning. Then the two step-200 losses **with the seed spread printed beside them**, the LR each schedule was at when stopped, the WSD-decay-from-200 result, and the keep decision in one clear sentence with its reason. Quote `gap_exceeds_seed_noise` explicitly; if it is `false`, the headline is "no measurable difference at step 200", and the keep decision rests on optionality alone. Add the one-line note that reading step 200 out of the 300-step run is identical to stopping there, because the schedule never looks past the current step.

- [ ] **Step 6: Fill Q5**

Embed `results/lr_sweep.png`. Table of the three minima (grid argmin, parabola-refined, `n_diverged`, `min_is_interior`). Report the fitted slope and what it means (`factor_per_width_doubling` = how much the optimal LR moves per doubling of width). Give the width-4096 prediction, the leave-one-out interval, and an explicit confidence statement: **low**, because three points, one seed, 200 steps, one depth, and an extrapolation two doublings past the data. Name the `n_heads`-fixed confound from `model.py` here — head_dim scales with width in this setup, so `lr*` absorbs both effects. Say what would raise confidence: a confirmation run at 2048, repeats at 2–3 seeds, or switching to µP so `lr*` is width-invariant and no extrapolation is needed at all.

- [ ] **Step 7: Fill "How to reproduce"**

````markdown
```bash
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests/ -v
python -m experiments.exp1_adam_by_hand
python -m experiments.exp2_bias_correction
python -m experiments.exp3_update_ratio
python -m experiments.exp4_cosine_vs_wsd
python -m experiments.exp5_lr_sweep
pip freeze > results/environment.txt
```
````
Include measured wall-clock time per script and the hardware you ran on. Commit `results/environment.txt` — `torch>=2.1` is a floor, not a pin, and a grader on a newer build will not reproduce your loss numbers to the digit without knowing which version produced them. Name the exact torch version in the README's setup section.

- [ ] **Step 8: Verify every number against `results/`**

Run: `cat results/*.json`
Go line by line through the README. Any number not present in a results file is either a typo or a fabrication — fix it or delete the claim.

- [ ] **Step 9: Commit**

```bash
git add README.md
git commit -m "docs(s11): README with all five answers, figures and reproduction steps"
```

---

## Task 11: Submission

**Files:**
- Modify: `README.md` (final polish only)

- [ ] **Step 1: Full green run from a clean checkout**

```bash
git clone <your-repo-url> /tmp/s11check && cd /tmp/s11check && pip install -r requirements.txt && python -m pytest tests/ -v
```
Expected: all tests pass on a machine that has never seen your working tree. If an import fails, you have a path assumption to fix.

- [ ] **Step 2: Confirm every figure is committed and renders**

Run: `git ls-files results | grep png`
Expected: 4 PNGs (`bias_correction`, `update_ratio`, `schedules`, `lr_sweep`). Open the rendered README on GitHub and confirm each image loads — relative paths like `results/lr_sweep.png` work; absolute local paths do not.

- [ ] **Step 3: Push and verify public access in an incognito window**

```bash
git push origin main
```
Then open the repo URL in a private/incognito window. The submission form has a checkbox asserting exactly this — do not tick it until you have actually done it.

- [ ] **Step 4: Submit**

Paste the direct link to `README.md` into the "GitHub README.md" field, add a short caption (e.g. "Session 11 — Adam by hand, bias correction, warmup ratios, cosine vs WSD, LR-vs-width"), tick the incognito box, and submit. Resubmission is allowed, so submit as soon as the README is complete rather than holding it back for polish.

---

## Self-Review

**Revision note:** this plan was audited against the brief after first drafting. Six defects were found and patched — recorded here so an executor knows why the code looks the way it does and does not "simplify" the fixes back out.

| # | Defect | Fix |
|---|---|---|
| 1 | Q1 compared only the final weight; the brief says check **each** of m/v/m̂/v̂ | Task 3 now pulls `exp_avg`/`exp_avg_sq` from torch's optimizer state and reports `max_diff_by_quantity` |
| 2 | `parabola_min` could fit through the `1e9` divergence sentinel and silently return a meaningless `lr*` | Task 9 filters to finite losses before fitting; `parabola_min` asserts it |
| 3 | Seed-variance check was prose with no code — the plan's own no-placeholder rule | Task 8 runs 3 seeds per schedule and emits `seed_spread` / `gap_exceeds_seed_noise` |
| 4 | `stop_at` was a dead parameter on `train_run` | Removed, with a docstring explaining why truncation is unnecessary |
| 5 | `n_heads` fixed while width varies — an unnamed confound in Q5 | Documented in `model.py` and required in the README's Q5 section |
| 6 | `torch>=2.1` is a floor, not a pin — loss numbers not reproducible across versions | `pip freeze > results/environment.txt`, committed and cited |

**Spec coverage:**
- Q1 → Task 2 (implementation + torch agreement test) and Task 3 (five-step table, all five quantities checked against PyTorch). ✅
- Q2 → Task 4 (20-step plot both ways, two stated definitions of "stops mattering"). ✅
- Q3 → Task 5 (`UpdateRatioLogger`) and Task 7 (per-layer plot + programmatic knee). ✅
- Q4 → Task 8 (300 steps, cosine and WSD, both tuned, both read at step 200, seed spread, keep decision). ✅
- Q5 → Task 9 (widths 256/512/1024, loss-vs-LR plot, three minima marked, 4096 prediction with confidence and named confound). ✅
- "Tune both sides" warning → Task 8 Step 1 `tune()` sweeps peak LR per schedule, and the seed spread stops an untuned-noise result being read as a win; Global Constraints makes both non-negotiable. ✅
- README + support code deliverable → Tasks 10 and 11. ✅

**Placeholder scan:** no TBDs; every code step carries runnable code; every "add error handling"-style instruction has been replaced with the actual check to run.

**Type consistency:** `train_run` is called with the same keyword names in Tasks 6, 7, 8, 9. `schedule_fn` is uniformly `(step, total_steps) -> float`. `UpdateRatioLogger.rows` dict keys (`step, layer, lr, w_rms, d_rms, ratio`) match the CSV writer in Task 7 and the test assertions in Task 5. `RunResult.loss_at` is defined in Task 6 and used in Task 8. `new_fig`/`finish` are defined in Task 4 and used in Tasks 7, 8, 9.

**Known risks:**
- Task 9 is the compute bottleneck (21 runs, width up to 1024). If CPU-only, halve `seq_len` to 64 and `batch_size` to 8 *before* the first run and record it in the README config table — never mid-sweep, because that invalidates the cross-width comparison.
- Task 8 is now 19 runs rather than 13, because of the seed-spread addition. That is the right trade: an untested comparison is worth less than no comparison. Do not cut it to save time.
- Total compute across Tasks 7–9 is roughly 43 training runs. On CPU budget 1.5–2.5 hours; on a T4, ~15 minutes. Run Tasks 1–4 first — they finish in seconds, need no GPU, and answer two of the five questions outright.

**Total budget check:** Tasks 1–6 are minutes. Tasks 7–9 are the compute. Task 10 (the README) is where the marks are and deserves a real sitting, not the last twenty minutes.
