# Reversible 20M-LLM Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a ~20M-parameter GPT on 50M tokens three times (baseline at a fixed batch size → reversible at the same batch size → reversible at the maximum batch size), and publish a public GitHub repo with the executed notebooks and a README that reports final loss, tokens/s, peak memory and findings.

**Architecture:** A small Python package `revllm` holds all logic (data, model, reversible blocks with a memory-saving custom autograd function, training loop with metrics, batch-size search, report/plots, notebook generator), fully unit-tested on CPU. Thin Colab notebooks import the package, run on a T4 GPU, and write one JSON result file per run to Google Drive. A local analysis notebook turns the JSONs into the README table and plots.

**Tech Stack:** Python 3.10+, PyTorch ≥ 2.4 (SDPA attention, `torch.amp`), tiktoken (GPT-2 BPE), HuggingFace `datasets` (TinyStories), numpy memmap, matplotlib, nbformat, pytest, Google Colab (T4, 16 GB).

**Spec:** The assignment text (reproduced verbatim below). No separate design doc.

> Train a 20M LLM for 50M tokens on Google Colab (or anything else of your choice). Fix Batch size that you can run. Train again with Reversibility (report which variant worked for you, mid-point, euler, etc) Train again with Reversibility, but push it to the maximum batch size.
> Report final loss, speed (token/s), memory peak and other findings. Submit detailed README.md (github link), repo must have the ipynb notebooks.

## Global Constraints

- Model size: ~20M parameters. Target config `d_model=256, n_layer=8, n_head=8, block_size=512, vocab=50257`, tied embeddings, which gives ≈ 19.3M.
- Token budget: exactly `50_000_000` tokens per full run (`TrainConfig.total_tokens`). `tokens_seen` is recorded in every result JSON.
- Hardware: Google Colab free tier, **T4 GPU** (sm_75, so fp16 + GradScaler; bf16 is used automatically on sm_80+).
- The same data, optimizer, LR schedule, seed and chunked loss are used for **all** runs. Only `arch`, `h`, `batch_size` and (for the max-batch run) `lr` may differ.
- Baseline batch size is fixed at **64** (32,768 tokens/step), provided the batch search confirms it fits.
- The repo must be **public** on GitHub and contain `notebooks/*.ipynb` **with outputs**, plus `README.md`.
- `torch>=2.4` (needed for `torch.amp.custom_fwd(device_type=...)`).
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

---

## How Claude skills are used in this project

| Phase | Skill / tool | What it does here |
|---|---|---|
| Plan | `superpowers:writing-plans` | Produced this document. |
| (Optional) re-scope | `superpowers:brainstorming` | Only if you want to change a design decision below (dataset, GPU, variants). |
| Execute tasks 1–7 | `superpowers:subagent-driven-development` (or `superpowers:executing-plans`) | One fresh subagent per task, reviewed between tasks. |
| Inside every code task | `superpowers:test-driven-development` | Failing test → minimal code → passing test → commit. |
| When anything breaks (grad mismatch, NaN, OOM) | `superpowers:systematic-debugging` | Root-cause before patching. |
| Plots (Task 6, Task 10) | `dataviz` | Chart form, colors and labeling for the README figures. |
| After Task 7 and after Task 10 | `code-review` / `superpowers:requesting-code-review` | Review the diff before publishing. |
| Before saying "done" | `superpowers:verification-before-completion` | Re-run tests, check the numbers in the README against the JSONs, and check the public link. |
| Publishing | `superpowers:finishing-a-development-branch` | Final commit and push to GitHub. |
| Public-link check | Built-in browser pane (isolated from your Chrome logins) | Loads the README URL logged-out. You still do the incognito check and tick the box yourself. |

## Assignment checkpoint → where it is satisfied

| Checkpoint | Satisfied by |
|---|---|
| 20M LLM | Task 2 (`test_param_count_is_about_20M`), `ModelConfig` defaults |
| 50M tokens | `TrainConfig.total_tokens=50_000_000`, `tokens_seen` in JSON (Task 4) |
| On Google Colab | Notebooks 01–03 (Task 7), run in Task 9 |
| Fix a batch size you can run | Notebook 01: B=64 verified by `fits()`; baseline max batch also reported |
| Train again with reversibility | Task 3 (reversible blocks + O(1)-activation backward), notebook 02 full run at B=64 |
| Report which variant worked (midpoint, euler…) | Notebook 02 pilots: `euler h=1`, `midpoint h=0.5`, `midpoint h=0.25` (+ baseline reference). README "Variants" section |
| Reversible at maximum batch size | Task 5 (`find_max_batch`), notebook 03 |
| Final loss, tokens/s, peak memory | Task 4 metrics, Task 6 table |
| Other findings | Reconstruction error, divergence, speed overhead, large-batch/equal-token effect, logits-memory note (README) |
| Detailed README + GitHub link | Task 10, Task 11 |
| Repo has the ipynb notebooks | Task 9 (executed notebooks committed) |
| Publicly accessible (incognito) | Task 11 |

## Key design decisions (read before executing)

1. **Chunked LM loss for all runs.** With a 50,257-token vocabulary, full fp32 logits cost about 300 KB/token, which is roughly 5× more than all 8 transformer blocks combined (about 70 KB/token). Reversibility only removes block activations, so without this change it would save almost nothing. `chunked_lm_loss` computes the head and cross-entropy in 2,048-token checkpointed chunks. It is applied identically to the baseline and reversible runs, so the comparison still isolates the effect of reversibility. The README states this.
2. **Variants.**
   - `euler`: two-stream additive coupling (RevNet/Reformer, which is a symplectic-Euler step). The update is `y1 = x1 + h·Attn(LN x2)`, then `y2 = x2 + h·MLP(LN y1)`.
   - `midpoint`: leapfrog / explicit-midpoint step on the states `(x_{n-1}, x_n)`. The update is `(a, b) → (b, a + 2h·F(b))`, where `F` is a full pre-LN transformer block delta. With `h=0.5`, the first step equals a standard residual block.
3. **Memory saving.** A custom `torch.autograd.Function` runs all reversible blocks under `no_grad` and saves **only the final two streams**. In backward, each block's input is reconstructed with `inverse()`, recomputed with grad enabled, and differentiated locally.
4. **Speed metric.** `tokens_per_sec` = tokens / GPU time for forward, backward and optimizer steps, excluding the first 10 steps and data loading. Wall-clock time is also stored.
5. **Memory metric.** `torch.cuda.max_memory_allocated()` after `reset_peak_memory_stats()` at train start. Reserved memory is also stored.

## File structure

```
reversible-llm/
├── pyproject.toml
├── .gitignore
├── README.md                          # Task 10
├── src/revllm/
│   ├── __init__.py
│   ├── config.py                      # ModelConfig, TrainConfig
│   ├── data.py                        # TinyStories → uint16 memmap, get_batch
│   ├── layers.py                      # attention, MLP, branches, standard Block
│   ├── reversible.py                  # Euler/Midpoint rev blocks, _RevStack autograd fn, recon error
│   ├── model.py                       # GPT, chunked_lm_loss
│   ├── train.py                       # train loop + metrics → results/<run>.json
│   ├── batch_search.py                # fits(), find_max_batch()
│   ├── report.py                      # results table + plots
│   └── notebooks.py                   # generates notebooks/*.ipynb
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_reversible.py
│   ├── test_train.py
│   ├── test_batch_search.py
│   ├── test_report.py
│   └── test_notebooks.py
├── notebooks/                         # generated (Task 7), executed on Colab (Task 9)
│   ├── 01_baseline.ipynb
│   ├── 02_reversible_same_batch.ipynb
│   ├── 03_reversible_max_batch.ipynb
│   └── 04_analysis.ipynb
├── results/                           # JSONs from Colab + plots/
└── docs/superpowers/plans/2026-09-26-reversible-llm.md
```

---

### Task 1: Repo scaffold, config, data pipeline

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `src/revllm/__init__.py`, `src/revllm/config.py`, `src/revllm/data.py`
- Test: `tests/conftest.py`, `tests/test_data.py`

**Interfaces:**
- Produces: `ModelConfig(vocab_size, block_size, n_layer, n_head, d_model, arch, h, chunked_loss, loss_chunk_tokens)`, `TrainConfig(...)` (fields below), `prepare_tinystories(data_dir, train_tokens=55_000_000, val_tokens=1_000_000) -> None`, `load_split(data_dir, split) -> np.memmap`, `get_batch(data, batch_size, block_size, device, generator=None) -> (x, y)` (int64 tensors, shape `(B, T)`), and the pytest fixture `tiny_cfg(arch="baseline", **kw) -> ModelConfig`.

- [ ] **Step 1: Scaffold the repo and environment**

```bash
cd /c/adi-python/reversible-llm && git init -b main
```

`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "revllm"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = ["torch>=2.4", "numpy", "tiktoken", "datasets", "matplotlib"]

[project.optional-dependencies]
dev = ["pytest", "nbformat", "nbconvert", "ipykernel"]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
data/
*.bin
*.egg-info/
.ipynb_checkpoints/
```

`src/revllm/__init__.py`: empty file.

```bash
cd /c/adi-python/reversible-llm && python -m venv .venv && .venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/Scripts/python -m pip install -e ".[dev]"
```

- [ ] **Step 2: Write the failing test**

`tests/conftest.py`:
```python
import pytest

from revllm.config import ModelConfig


@pytest.fixture
def tiny_cfg():
    def make(arch="baseline", **kw):
        return ModelConfig(vocab_size=97, block_size=16, n_layer=3, n_head=2, d_model=32, arch=arch, **kw)
    return make
```

`tests/test_data.py`:
```python
import numpy as np
import torch

from revllm.config import ModelConfig
from revllm.data import get_batch, load_split


def test_get_batch_targets_are_inputs_shifted_by_one(tmp_path):
    np.arange(1000, dtype=np.uint16).tofile(tmp_path / "train.bin")
    data = load_split(str(tmp_path), "train")
    x, y = get_batch(data, 4, 8, "cpu", torch.Generator().manual_seed(0))
    assert x.shape == (4, 8) and x.dtype == torch.int64
    assert torch.equal(y, x + 1)


def test_step_size_defaults_depend_on_variant():
    assert ModelConfig(arch="euler").h == 1.0
    assert ModelConfig(arch="midpoint").h == 0.5
    assert ModelConfig(arch="midpoint", h=0.25).h == 0.25
```

- [ ] **Step 3: Run it to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revllm.config'`

- [ ] **Step 4: Implement**

`src/revllm/config.py`:
```python
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 256
    arch: str = "baseline"          # "baseline" | "euler" | "midpoint"
    h: float | None = None          # reversible step size; None -> variant default
    chunked_loss: bool = True       # compute LM head + CE in checkpointed chunks (all runs)
    loss_chunk_tokens: int = 2048

    def __post_init__(self):
        if self.h is None:
            self.h = 0.5 if self.arch == "midpoint" else 1.0


@dataclass
class TrainConfig:
    run_name: str = "baseline"
    total_tokens: int = 50_000_000
    batch_size: int = 64
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every_tokens: int = 500_000
    eval_every_tokens: int = 2_500_000
    eval_tokens: int = 500_000
    seed: int = 1337
    data_dir: str = "data"
    out_dir: str = "results"
```

`src/revllm/data.py`:
```python
import os

import numpy as np
import torch


def prepare_tinystories(data_dir: str, train_tokens: int = 55_000_000, val_tokens: int = 1_000_000) -> None:
    """Tokenize TinyStories with GPT-2 BPE into uint16 memmaps. Skips splits that already exist."""
    import tiktoken
    from datasets import load_dataset

    os.makedirs(data_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    for split, hf_split, n_target in (("train", "train", train_tokens), ("val", "validation", val_tokens)):
        path = os.path.join(data_dir, f"{split}.bin")
        if os.path.exists(path):
            continue
        buf = np.empty(n_target, dtype=np.uint16)
        pos = 0
        for ex in load_dataset("roneneldan/TinyStories", split=hf_split, streaming=True):
            ids = enc.encode_ordinary(ex["text"]) + [enc.eot_token]
            take = min(len(ids), n_target - pos)
            buf[pos:pos + take] = ids[:take]
            pos += take
            if pos >= n_target:
                break
        buf[:pos].tofile(path)
        print(f"{split}: {pos:,} tokens -> {path}")


def load_split(data_dir: str, split: str) -> np.memmap:
    return np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")


def get_batch(data, batch_size: int, block_size: int, device: str, generator=None):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator).tolist()
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_data.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: scaffold revllm package with configs and TinyStories data pipeline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Baseline GPT model with chunked loss

**Files:**
- Create: `src/revllm/layers.py`, `src/revllm/model.py`
- Create (stub so that `model.py` imports): `src/revllm/reversible.py`, containing only the `make_rev_blocks` and `rev_forward` names. Task 3 replaces it.
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `ModelConfig`, `tiny_cfg`.
- Produces: `layers.AttnBranch(cfg)`, `layers.MLPBranch(cfg)`, `layers.Block(cfg)` with `.delta(x)` and `.forward(x) = x + delta(x)`; `model.GPT(cfg)` with `.embed(idx)`, `.head(x)`, `.num_params()`, `.forward(idx, targets=None, use_rev_fn=True) -> (logits | None, loss | None)`; `model.chunked_lm_loss(h, targets, head, chunk_tokens) -> scalar`.

- [ ] **Step 1: Write the failing test**

`tests/test_model.py`:
```python
import math

import torch

from revllm.config import ModelConfig
from revllm.model import GPT


def test_param_count_is_about_20M():
    n = GPT(ModelConfig()).num_params()
    assert 18e6 < n < 21e6, n


def test_initial_loss_is_near_uniform(tiny_cfg):
    torch.manual_seed(0)
    m = GPT(tiny_cfg())
    x = torch.randint(97, (2, 16))
    _, loss = m(x, x)
    assert abs(loss.item() - math.log(97)) < 0.3


def test_forward_without_targets_returns_logits(tiny_cfg):
    m = GPT(tiny_cfg())
    logits, loss = m(torch.randint(97, (2, 16)))
    assert logits.shape == (2, 16, 97) and loss is None


def test_chunked_loss_matches_full_loss_and_grads(tiny_cfg):
    torch.manual_seed(0)
    m = GPT(tiny_cfg(loss_chunk_tokens=5)).double()
    x, y = torch.randint(97, (3, 16)), torch.randint(97, (3, 16))
    _, l_chunked = m(x, y)
    l_chunked.backward()
    g_chunked = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    m.cfg.chunked_loss = False
    _, l_full = m(x, y)
    l_full.backward()
    assert torch.allclose(l_chunked, l_full, rtol=1e-10)
    for g, p in zip(g_chunked, m.parameters()):
        assert torch.allclose(g, p.grad, rtol=1e-8, atol=1e-12)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revllm.model'`

- [ ] **Step 3: Implement**

`src/revllm/layers.py`:
```python
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=False)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        return self.proj(F.gelu(self.fc(x)))


class AttnBranch(nn.Module):
    """f(x) = Attn(LN(x))"""
    def __init__(self, cfg):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)

    def forward(self, x):
        return self.attn(self.ln(x))


class MLPBranch(nn.Module):
    """g(x) = MLP(LN(x))"""
    def __init__(self, cfg):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        return self.mlp(self.ln(x))


class Block(nn.Module):
    """Standard pre-LN transformer block: x + F(x)."""
    def __init__(self, cfg):
        super().__init__()
        self.f = AttnBranch(cfg)
        self.g = MLPBranch(cfg)

    def delta(self, x):
        a = self.f(x)
        return a + self.g(x + a)

    def forward(self, x):
        return x + self.delta(x)
```

`src/revllm/reversible.py` (temporary stub, replaced in Task 3):
```python
def make_rev_blocks(cfg):
    raise NotImplementedError


def rev_forward(blocks, x, arch, memory_saving=True):
    raise NotImplementedError
```

`src/revllm/model.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .layers import Block
from .reversible import make_rev_blocks, rev_forward


def _ce_sum(h, t, head):
    return F.cross_entropy(head(h).float(), t, reduction="sum")


def chunked_lm_loss(h, targets, head, chunk_tokens):
    """Mean CE computed chunk-by-chunk; each chunk's logits are recomputed in backward,
    so peak logits memory is O(chunk_tokens * vocab) instead of O(B * T * vocab)."""
    h = h.reshape(-1, h.size(-1))
    t = targets.reshape(-1)
    total = h.new_zeros((), dtype=torch.float32 if h.dtype != torch.float64 else torch.float64)
    for i in range(0, h.size(0), chunk_tokens):
        total = total + checkpoint(_ce_sum, h[i:i + chunk_tokens], t[i:i + chunk_tokens], head, use_reentrant=False)
    return total / t.numel()


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.wpe = nn.Embedding(cfg.block_size, cfg.d_model)
        if cfg.arch == "baseline":
            self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        else:
            self.blocks = make_rev_blocks(cfg)
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight  # tied embeddings
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    def embed(self, idx):
        return self.wte(idx) + self.wpe(torch.arange(idx.size(1), device=idx.device))

    def head(self, x):
        return self.lm_head(self.ln_f(x))

    def forward(self, idx, targets=None, use_rev_fn=True):
        x = self.embed(idx)
        if self.cfg.arch == "baseline":
            for blk in self.blocks:
                x = blk(x)
        else:
            x = rev_forward(self.blocks, x, self.cfg.arch, memory_saving=use_rev_fn)
        if targets is None:
            return self.head(x), None
        if self.cfg.chunked_loss:
            return None, chunked_lm_loss(x, targets, self.head, self.cfg.loss_chunk_tokens)
        logits = self.head(x)
        return logits, F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_model.py -v`
Expected: 4 passed. The parameter count should be about 19.3M. If it isn't, fix the config rather than the test.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add 19.3M-param GPT baseline with chunked LM loss

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Reversible blocks (Euler, Midpoint) + memory-saving backward

**Files:**
- Modify: `src/revllm/reversible.py` (replace the stub entirely)
- Test: `tests/test_reversible.py`

**Interfaces:**
- Consumes: `layers.AttnBranch`, `layers.MLPBranch`, `layers.Block`, `GPT(cfg).forward(..., use_rev_fn=...)`, `GPT.embed`.
- Produces: `EulerRevBlock(cfg)`, `MidpointRevBlock(cfg)` (each has `forward(x1, x2) -> (y1, y2)` and `inverse(y1, y2) -> (x1, x2)`), `make_rev_blocks(cfg) -> nn.ModuleList`, `rev_forward(blocks, x, arch, memory_saving=True) -> Tensor`, `reconstruction_error(blocks, x) -> float`.

- [ ] **Step 1: Write the failing test**

`tests/test_reversible.py`:
```python
import pytest
import torch

from revllm.config import ModelConfig
from revllm.model import GPT
from revllm.reversible import make_rev_blocks, reconstruction_error

VARIANTS = ["euler", "midpoint"]


@pytest.mark.parametrize("arch", VARIANTS)
def test_block_inverse_reconstructs_inputs(tiny_cfg, arch):
    torch.manual_seed(0)
    blk = make_rev_blocks(tiny_cfg(arch))[0].double()
    x1 = torch.randn(2, 16, 32, dtype=torch.float64)
    x2 = torch.randn(2, 16, 32, dtype=torch.float64)
    r1, r2 = blk.inverse(*blk(x1, x2))
    assert torch.allclose(r1, x1, atol=1e-10) and torch.allclose(r2, x2, atol=1e-10)


@pytest.mark.parametrize("arch", VARIANTS)
def test_memory_saving_backward_matches_plain_autograd(tiny_cfg, arch):
    torch.manual_seed(0)
    m = GPT(tiny_cfg(arch)).double()
    x, y = torch.randint(97, (2, 16)), torch.randint(97, (2, 16))
    _, l_ref = m(x, y, use_rev_fn=False)
    l_ref.backward()
    ref = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    _, l_rev = m(x, y, use_rev_fn=True)
    l_rev.backward()
    assert torch.allclose(l_ref, l_rev, rtol=1e-12)
    for (name, p), g in zip(m.named_parameters(), ref):
        assert torch.allclose(p.grad, g, rtol=1e-6, atol=1e-10), name


@pytest.mark.parametrize("arch", VARIANTS)
def test_reconstruction_error_is_tiny_in_fp64(tiny_cfg, arch):
    torch.manual_seed(0)
    m = GPT(tiny_cfg(arch)).double()
    assert reconstruction_error(m.blocks, torch.randn(2, 16, 32, dtype=torch.float64)) < 1e-9


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("arch", VARIANTS)
def test_reversible_uses_less_activation_memory(arch):
    cfg = ModelConfig(arch=arch, vocab_size=512)  # small vocab so block activations dominate

    def activation_peak(use_rev):
        torch.manual_seed(0)
        m = GPT(cfg).cuda()
        x = torch.randint(cfg.vocab_size, (8, cfg.block_size), device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        _, loss = m(x, x, use_rev_fn=use_rev)
        loss.backward()
        return torch.cuda.max_memory_allocated() - base

    assert activation_peak(True) < 0.5 * activation_peak(False)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_reversible.py -v`
Expected: FAIL with `ImportError: cannot import name 'reconstruction_error'`

- [ ] **Step 3: Implement**

`src/revllm/reversible.py`:
```python
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
def reconstruction_error(blocks, x) -> float:
    """Forward through all blocks, then invert back down the stack (as backward does).
    Returns the max |true input - reconstructed input| over all layers."""
    states = [(x, x)]
    for blk in blocks:
        states.append(blk(*states[-1]))
    y1, y2 = states[-1]
    err = 0.0
    for i in range(len(blocks) - 1, -1, -1):
        y1, y2 = blocks[i].inverse(y1, y2)
        err = max(err, (y1 - states[i][0]).abs().max().item(), (y2 - states[i][1]).abs().max().item())
    return err
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: all pass. The CUDA memory test is skipped locally and runs on Colab in Task 9. If the gradients mismatch, use `superpowers:systematic-debugging` and check the per-block gradient ordering in `backward`.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add Euler and midpoint reversible blocks with activation-free backward

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Training loop with metrics

**Files:**
- Create: `src/revllm/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `ModelConfig`, `TrainConfig`, `load_split`, `get_batch`, `GPT`, `reconstruction_error`.
- Produces: `lr_at(step, total_steps, tc) -> float`, `amp_dtype() -> torch.dtype`, and `train(mc, tc, max_steps=None) -> dict`, which also writes `<tc.out_dir>/<tc.run_name>.json`. Result keys: `run_name, arch, h, batch_size, params, steps, tokens_seen, diverged, final_train_loss, final_val_loss, tokens_per_sec, wall_time_s, peak_mem_allocated_gb, peak_mem_reserved_gb, gpu, amp_dtype, recon_error, model_config, train_config, history{tokens, train_loss, val_tokens, val_loss}`.

- [ ] **Step 1: Write the failing test**

`tests/test_train.py`:
```python
import json

import numpy as np
import pytest

from revllm.config import TrainConfig
from revllm.train import lr_at, train


def test_lr_schedule_warmup_then_cosine():
    tc = TrainConfig(lr=1e-3, min_lr=1e-4, warmup_steps=10)
    assert lr_at(0, 100, tc) == pytest.approx(1e-4)
    assert lr_at(9, 100, tc) == pytest.approx(1e-3)
    assert lr_at(99, 100, tc) == pytest.approx(1e-4, abs=1e-5)
    assert lr_at(1, 20, tc) == pytest.approx(1e-3)  # warmup capped at total_steps // 10


@pytest.mark.parametrize("arch", ["baseline", "euler", "midpoint"])
def test_train_smoke_learns_and_writes_json(tmp_path, tiny_cfg, arch):
    pattern = np.tile(np.arange(97, dtype=np.uint16), 300)  # next token = current + 1
    pattern.tofile(tmp_path / "train.bin")
    pattern[:3000].tofile(tmp_path / "val.bin")
    tc = TrainConfig(run_name=f"smoke_{arch}", batch_size=8, lr=3e-3, min_lr=3e-4, warmup_steps=5,
                     log_every_tokens=128, eval_every_tokens=2048, eval_tokens=256,
                     data_dir=str(tmp_path), out_dir=str(tmp_path / "results"))
    r = train(tiny_cfg(arch), tc, max_steps=80)
    assert not r["diverged"]
    assert r["history"]["train_loss"][-1] < r["history"]["train_loss"][0] - 0.5
    saved = json.loads((tmp_path / "results" / f"smoke_{arch}.json").read_text())
    assert saved["steps"] == 80 and saved["tokens_seen"] == 80 * 8 * 16
    assert saved["tokens_per_sec"] > 0
    if arch != "baseline":
        assert r["recon_error"] < 1e-3
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revllm.train'`

- [ ] **Step 3: Implement**

`src/revllm/train.py`:
```python
import json
import math
import os
import time
from dataclasses import asdict

import torch

from .config import ModelConfig, TrainConfig
from .data import get_batch, load_split
from .model import GPT
from .reversible import reconstruction_error


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
    n_batches = max(1, tc.eval_tokens // (tc.batch_size * mc.block_size))
    total = 0.0
    for _ in range(n_batches):
        x, y = get_batch(val_data, tc.batch_size, mc.block_size, device, gen)
        with torch.autocast(device_type=device, dtype=dtype, enabled=device == "cuda"):
            _, loss = model(x, y)
        total += loss.item()
    model.train()
    return total / n_batches


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
    history = {"tokens": [], "train_loss": [], "val_tokens": [], "val_loss": []}
    gen = torch.Generator().manual_seed(tc.seed)
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    diverged, steps_done = False, 0
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
        scaler.update()
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
            history["val_tokens"].append(tokens)
            history["val_loss"].append(evaluate(model, val_data, mc, tc, device, dtype))

    if not diverged:
        history["val_tokens"].append(steps_done * tokens_per_step)
        history["val_loss"].append(evaluate(model, val_data, mc, tc, device, dtype))

    recon = None
    if mc.arch != "baseline" and not diverged:
        x, _ = get_batch(val_data, 4, mc.block_size, device, torch.Generator().manual_seed(1))
        with torch.no_grad(), torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            recon = reconstruction_error(model.blocks, model.embed(x))

    nan = float("nan")
    result = {
        "run_name": tc.run_name, "arch": mc.arch, "h": mc.h, "batch_size": tc.batch_size,
        "params": model.num_params(), "steps": steps_done, "tokens_seen": steps_done * tokens_per_step,
        "diverged": diverged,
        "final_train_loss": nan if diverged else sum(history["train_loss"][-5:]) / len(history["train_loss"][-5:]),
        "final_val_loss": nan if diverged else history["val_loss"][-1],
        "tokens_per_sec": timed_tokens / timed_seconds if timed_seconds else None,
        "wall_time_s": time.time() - t_start,
        "peak_mem_allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None,
        "peak_mem_reserved_gb": torch.cuda.max_memory_reserved() / 1e9 if device == "cuda" else None,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "amp_dtype": str(dtype) if use_amp else "float32",
        "recon_error": recon,
        "model_config": asdict(mc), "train_config": asdict(tc), "history": history,
    }
    os.makedirs(tc.out_dir, exist_ok=True)
    with open(os.path.join(tc.out_dir, f"{tc.run_name}.json"), "w") as f:
        json.dump(result, f, indent=2)
    return result
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: all pass. The CUDA test is skipped.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add training loop reporting loss, tokens/s, peak memory, recon error

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Maximum batch-size search

**Files:**
- Create: `src/revllm/batch_search.py`
- Test: `tests/test_batch_search.py`

**Interfaces:**
- Consumes: `GPT`, `ModelConfig`, `amp_dtype`.
- Produces: `fits(mc, batch_size, steps=2) -> bool` (CUDA only) and `find_max_batch(mc, start=8, limit=4096, safety=0.9) -> {"arch", "max_batch", "recommended", "first_oom"}`.

- [ ] **Step 1: Write the failing test**

`tests/test_batch_search.py`:
```python
import pytest

from revllm import batch_search
from revllm.config import ModelConfig


def test_finds_exact_boundary_and_safe_recommendation(monkeypatch):
    monkeypatch.setattr(batch_search, "fits", lambda mc, b, steps=2: b <= 100)
    r = batch_search.find_max_batch(ModelConfig())
    assert r["max_batch"] == 100 and r["first_oom"] == 101
    assert r["recommended"] == 88  # 90% of 100, rounded down to a multiple of 8


def test_respects_limit(monkeypatch):
    monkeypatch.setattr(batch_search, "fits", lambda mc, b, steps=2: True)
    assert batch_search.find_max_batch(ModelConfig(), limit=64)["max_batch"] == 64


def test_raises_when_nothing_fits(monkeypatch):
    monkeypatch.setattr(batch_search, "fits", lambda mc, b, steps=2: False)
    with pytest.raises(RuntimeError):
        batch_search.find_max_batch(ModelConfig())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_batch_search.py -v`
Expected: FAIL with `ImportError: cannot import name 'batch_search'`

- [ ] **Step 3: Implement**

`src/revllm/batch_search.py`:
```python
import gc

import torch

from .config import ModelConfig
from .model import GPT
from .train import amp_dtype


def fits(mc: ModelConfig, batch_size: int, steps: int = 2) -> bool:
    """True if `steps` full train steps (fwd + bwd + AdamW) run at this batch size without OOM."""
    model = opt = None
    try:
        model = GPT(mc).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        for _ in range(steps):
            x = torch.randint(mc.vocab_size, (batch_size, mc.block_size), device="cuda")
            with torch.autocast("cuda", dtype=amp_dtype()):
                _, loss = model(x, x)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return True
    except torch.cuda.OutOfMemoryError:
        return False
    finally:
        del model, opt
        gc.collect()
        torch.cuda.empty_cache()


def find_max_batch(mc: ModelConfig, start: int = 8, limit: int = 4096, safety: float = 0.9) -> dict:
    """Double until OOM, then binary-search the exact boundary."""
    lo, hi = 0, start
    while hi <= limit and fits(mc, hi):
        lo, hi = hi, hi * 2
    hi = min(hi, limit + 1)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if fits(mc, mid):
            lo = mid
        else:
            hi = mid
    if lo == 0:
        raise RuntimeError(f"batch_size=1 does not fit for arch={mc.arch}")
    recommended = int(lo * safety) // 8 * 8 or max(1, int(lo * safety))
    return {"arch": mc.arch, "max_batch": lo, "recommended": recommended,
            "first_oom": hi if hi <= limit else None}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add OOM-boundary batch-size search

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Results table and plots

Load the `dataviz` skill before this task. It may adjust colors and labels, but the function names and outputs below must stay.

**Files:**
- Create: `src/revllm/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: result dicts from `train()` (the keys listed in Task 4).
- Produces: `load_results(results_dir) -> list[dict]` (sorted by `run_name`), `results_table(results) -> str` (Markdown), `plot_all(results, out_dir) -> list[str]` (4 PNG paths: `val_loss_full_runs.png`, `val_loss_pilots.png`, `peak_memory.png`, `throughput.png`).

- [ ] **Step 1: Write the failing test**

`tests/test_report.py`:
```python
import json
import os

from revllm.report import load_results, plot_all, results_table


def fake(name, arch, batch, val, diverged=False):
    return {"run_name": name, "arch": arch, "h": 1.0, "batch_size": batch, "tokens_seen": 50_000_000,
            "diverged": diverged, "final_train_loss": val + 0.05, "final_val_loss": val,
            "tokens_per_sec": 80000.0, "peak_mem_allocated_gb": 5.2, "wall_time_s": 660.0, "recon_error": None,
            "history": {"tokens": [1, 2], "train_loss": [5.0, val], "val_tokens": [2], "val_loss": [val]}}


def test_load_and_table(tmp_path):
    for r in (fake("02_pilot_midpoint_h0.5", "midpoint", 64, float("nan"), True), fake("01_baseline", "baseline", 64, 1.5)):
        (tmp_path / f"{r['run_name']}.json").write_text(json.dumps(r))
    rs = load_results(str(tmp_path))
    assert [r["run_name"] for r in rs] == ["01_baseline", "02_pilot_midpoint_h0.5"]
    table = results_table(rs)
    assert "| 01_baseline | baseline |" in table
    assert "diverged" in table and "1.500" in table


def test_plot_all_writes_four_pngs(tmp_path):
    paths = plot_all([fake("01_baseline", "baseline", 64, 1.5), fake("02_pilot_euler_h1.0", "euler", 64, 1.6)], str(tmp_path))
    assert len(paths) == 4 and all(os.path.getsize(p) > 0 for p in paths)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_report.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revllm.report'`

- [ ] **Step 3: Implement**

`src/revllm/report.py`:
```python
import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_results(results_dir: str) -> list[dict]:
    rs = []
    for p in glob.glob(os.path.join(results_dir, "*.json")):
        with open(p) as f:
            rs.append(json.load(f))
    return sorted(rs, key=lambda r: r["run_name"])


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:,.{nd}f}"


def results_table(results: list[dict]) -> str:
    rows = ["| Run | Arch | h | Batch | Tokens | Final train loss | Final val loss | Tokens/s | Peak mem (GB) | Wall (min) | Status |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        rows.append(
            f"| {r['run_name']} | {r['arch']} | {r['h']} | {r['batch_size']} | {r['tokens_seen'] / 1e6:.1f}M "
            f"| {_fmt(r['final_train_loss'])} | {_fmt(r['final_val_loss'])} | {_fmt(r['tokens_per_sec'], 0)} "
            f"| {_fmt(r['peak_mem_allocated_gb'], 2)} | {r['wall_time_s'] / 60:.1f} "
            f"| {'diverged' if r['diverged'] else 'ok'} |")
    return "\n".join(rows)


def _curves(results, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    for r in results:
        h = r["history"]
        ax.plot([t / 1e6 for t in h["val_tokens"]], h["val_loss"], marker="o", ms=3,
                label=f"{r['run_name']} (B={r['batch_size']})")
    ax.set(xlabel="Tokens seen (millions)", ylabel="Validation loss", title=title)
    ax.grid(alpha=0.3)
    if results:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _bars(results, key, label, out_path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    names = [r["run_name"] for r in results]
    vals = [r.get(key) or 0 for r in results]
    bars = ax.barh(names, vals)
    ax.bar_label(bars, fmt="%.2f" if vals and max(vals) < 100 else "%.0f", padding=3)
    ax.set(xlabel=label)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_all(results: list[dict], out_dir: str) -> list[str]:
    os.makedirs(out_dir, exist_ok=True)
    full = [r for r in results if "pilot" not in r["run_name"]]
    pilots = [r for r in results if "pilot" in r["run_name"]]
    j = lambda name: os.path.join(out_dir, name)  # noqa: E731
    return [
        _curves(full, j("val_loss_full_runs.png"), "Validation loss: full 50M-token runs"),
        _curves(pilots, j("val_loss_pilots.png"), "Validation loss: 5M-token variant pilots"),
        _bars(full, "peak_mem_allocated_gb", "Peak GPU memory allocated (GB)", j("peak_memory.png")),
        _bars(full, "tokens_per_sec", "Training throughput (tokens/s)", j("throughput.png")),
    ]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: add results table and comparison plots

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Notebook generator + the four notebooks

**Files:**
- Create: `src/revllm/notebooks.py`
- Generate: `notebooks/01_baseline.ipynb`, `notebooks/02_reversible_same_batch.ipynb`, `notebooks/03_reversible_max_batch.ipynb`, `notebooks/04_analysis.ipynb`
- Test: `tests/test_notebooks.py`

**Interfaces:**
- Consumes: `prepare_tinystories`, `ModelConfig`, `TrainConfig`, `train`, `find_max_batch`, `load_results`, `results_table`, `plot_all`.
- Produces: `build(repo_url, out_dir) -> list[Path]` and `colab_url(repo_url, notebook_name) -> str`. CLI: `python -m revllm.notebooks <repo_url>`, which writes into `notebooks/` and prints the Colab badge links.
- Colab ↔ Drive contract: results go to `/content/drive/MyDrive/reversible-llm/results/*.json`, the chosen variant to `.../winner.json`, and batch searches to `.../batch_search_<arch>.json`.

- [ ] **Step 1: Write the failing test**

`tests/test_notebooks.py`:
```python
import nbformat

from revllm.notebooks import build, colab_url

URL = "https://github.com/someone/reversible-llm.git"


def test_build_writes_four_valid_notebooks(tmp_path):
    paths = build(URL, tmp_path)
    assert sorted(p.name for p in paths) == ["01_baseline.ipynb", "02_reversible_same_batch.ipynb",
                                             "03_reversible_max_batch.ipynb", "04_analysis.ipynb"]
    for p in paths:
        nbformat.validate(nbformat.read(p, as_version=4))
    src = "".join(c.source for c in nbformat.read(tmp_path / "01_baseline.ipynb", as_version=4).cells)
    assert URL in src and "TrainConfig(" in src


def test_colab_url():
    assert colab_url(URL, "x.ipynb") == \
        "https://colab.research.google.com/github/someone/reversible-llm/blob/main/notebooks/x.ipynb"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_notebooks.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'revllm.notebooks'`

- [ ] **Step 3: Implement**

`src/revllm/notebooks.py`:
```python
"""Generates the Colab notebooks from source so they stay reviewable.
Usage: python -m revllm.notebooks https://github.com/<you>/reversible-llm.git"""
import sys
from pathlib import Path

import nbformat as nbf


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def colab_url(repo_url: str, name: str) -> str:
    slug = repo_url.removeprefix("https://github.com/").removesuffix(".git")
    return f"https://colab.research.google.com/github/{slug}/blob/main/notebooks/{name}"


def _setup(repo_url):
    return [
        md("## Setup\nRuntime → Change runtime type → **T4 GPU**. Results are written to Google Drive, "
           "so a disconnect does not lose finished runs."),
        code("!nvidia-smi"),
        code("""
from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/reversible-llm'
import os
os.makedirs(f'{DRIVE}/data', exist_ok=True)
os.makedirs(f'{DRIVE}/results', exist_ok=True)
"""),
        code(f"""
import os, sys
if not os.path.exists('/content/reversible-llm'):
    !git clone -q {repo_url} /content/reversible-llm
%cd /content/reversible-llm
!git pull -q
!pip install -q tiktoken datasets
sys.path.insert(0, '/content/reversible-llm/src')
"""),
        code("""
from revllm.data import prepare_tinystories
prepare_tinystories(f'{DRIVE}/data')  # tokenizes once (~55M tokens), cached on Drive
!mkdir -p /content/data && cp -n {DRIVE}/data/*.bin /content/data/
"""),
        code("!python -m pytest -q tests  # includes the GPU memory test that is skipped on CPU"),
    ]


def nb_baseline(repo_url):
    return [
        md("# 01: Baseline GPT (19.3M params, 50M tokens, fixed batch size)\n"
           "Standard pre-LN transformer. We first measure the largest batch that fits (for reporting), "
           "then train at a fixed **B=64** (32,768 tokens/step), a size that trains well within the 50M-token budget."),
        *_setup(repo_url),
        code("""
import json
from revllm.config import ModelConfig, TrainConfig
from revllm.batch_search import find_max_batch
from revllm.train import train

mc = ModelConfig(arch='baseline')
search = find_max_batch(mc)
json.dump(search, open(f'{DRIVE}/batch_search_baseline.json', 'w'))
search
"""),
        code("""
B = 64 if search['max_batch'] >= 64 else search['recommended']
res = train(mc, TrainConfig(run_name='01_baseline', batch_size=B,
                            data_dir='/content/data', out_dir=f'{DRIVE}/results'))
{k: res[k] for k in ['params', 'batch_size', 'tokens_seen', 'final_train_loss', 'final_val_loss',
                     'tokens_per_sec', 'peak_mem_allocated_gb', 'wall_time_s']}
"""),
    ]


def nb_same_batch(repo_url):
    return [
        md("# 02: Reversible transformer at the same batch size\n"
           "**Step 1: variant pilots.** Each variant (plus a baseline reference) is trained for 5M tokens at the baseline batch size:\n"
           "- `euler`: two-stream additive coupling (RevNet/Reformer, a symplectic-Euler step), h=1\n"
           "- `midpoint`: leapfrog/explicit-midpoint step `(a, b) → (b, a + 2h·F(b))`, h=0.5 and h=0.25\n\n"
           "**Step 2:** full 50M-token run of the variant with the best pilot validation loss."),
        *_setup(repo_url),
        code("""
import json
from revllm.config import ModelConfig, TrainConfig
from revllm.train import train

B = json.load(open(f'{DRIVE}/results/01_baseline.json'))['batch_size']
PILOTS = [('baseline', None), ('euler', 1.0), ('midpoint', 0.5), ('midpoint', 0.25)]
pilots = {}
for arch, h in PILOTS:
    mc = ModelConfig(arch=arch, h=h)
    name = f'02_pilot_{arch}_h{mc.h}'
    pilots[name] = train(mc, TrainConfig(run_name=name, total_tokens=5_000_000, batch_size=B,
                                         eval_every_tokens=1_000_000,
                                         data_dir='/content/data', out_dir=f'{DRIVE}/results'))
KEYS = ['diverged', 'final_val_loss', 'tokens_per_sec', 'peak_mem_allocated_gb', 'recon_error']
{n: {k: r[k] for k in KEYS} for n, r in pilots.items()}
"""),
        code("""
ok = [r for r in pilots.values() if r['arch'] != 'baseline' and not r['diverged']]
winner = min(ok, key=lambda r: r['final_val_loss'])
json.dump({'arch': winner['arch'], 'h': winner['h']}, open(f'{DRIVE}/winner.json', 'w'))
print('Winner:', winner['arch'], 'h =', winner['h'])
"""),
        code("""
mc = ModelConfig(arch=winner['arch'], h=winner['h'])
res = train(mc, TrainConfig(run_name=f"02_reversible_{winner['arch']}_same_batch", batch_size=B,
                            data_dir='/content/data', out_dir=f'{DRIVE}/results'))
{k: res[k] for k in ['batch_size', 'tokens_seen', 'final_train_loss', 'final_val_loss',
                     'tokens_per_sec', 'peak_mem_allocated_gb', 'wall_time_s', 'recon_error']}
"""),
    ]


def nb_max_batch(repo_url):
    return [
        md("# 03: Reversible transformer at the maximum batch size\n"
           "Search for the largest batch the winning variant can run, back off 10% for headroom, and train "
           "on the same 50M tokens. The learning rate is scaled by √(B/B_base) and capped at 3e-3."),
        *_setup(repo_url),
        code("""
import json, math
from revllm.config import ModelConfig, TrainConfig
from revllm.batch_search import find_max_batch
from revllm.train import train

w = json.load(open(f'{DRIVE}/winner.json'))
B_base = json.load(open(f'{DRIVE}/results/01_baseline.json'))['batch_size']
mc = ModelConfig(arch=w['arch'], h=w['h'])
search = find_max_batch(mc, start=B_base)
json.dump(search, open(f"{DRIVE}/batch_search_{w['arch']}.json", 'w'))
search
"""),
        code("""
B_max = search['recommended']
lr = min(3e-3, 1e-3 * math.sqrt(B_max / B_base))
res = train(mc, TrainConfig(run_name=f"03_reversible_{w['arch']}_max_batch", batch_size=B_max,
                            lr=lr, min_lr=lr / 10,
                            data_dir='/content/data', out_dir=f'{DRIVE}/results'))
{k: res[k] for k in ['batch_size', 'steps', 'tokens_seen', 'final_train_loss', 'final_val_loss',
                     'tokens_per_sec', 'peak_mem_allocated_gb', 'wall_time_s']}
"""),
    ]


def nb_analysis():
    return [
        md("# 04: Analysis\nRun locally (or in Colab) after copying the Drive `results/*.json` into the repo's `results/`."),
        code("""
import os
from pathlib import Path
root = Path.cwd() if (Path.cwd() / 'results').exists() else Path.cwd().parent
os.chdir(root)
import sys; sys.path.insert(0, str(root / 'src'))
from revllm.report import load_results, results_table, plot_all
from IPython.display import Image, Markdown, display

rs = load_results('results')
display(Markdown(results_table(rs)))
for p in plot_all(rs, 'results/plots'):
    display(Image(p))
"""),
    ]


def build(repo_url: str, out_dir) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = {
        "01_baseline.ipynb": nb_baseline(repo_url),
        "02_reversible_same_batch.ipynb": nb_same_batch(repo_url),
        "03_reversible_max_batch.ipynb": nb_max_batch(repo_url),
        "04_analysis.ipynb": nb_analysis(),
    }
    paths = []
    for name, cells in specs.items():
        nb = nbf.v4.new_notebook(cells=cells)
        nb.metadata["accelerator"] = "GPU"
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        path = out / name
        nbf.write(nb, path)
        paths.append(path)
    return paths


if __name__ == "__main__":
    url = sys.argv[1]
    for p in build(url, Path(__file__).resolve().parents[2] / "notebooks"):
        print(f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_url(url, p.name)}) `{p.name}`")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/ -v`
Expected: all pass.

- [ ] **Step 5: Code review gate**

Invoke the `code-review` skill (level `high`) on the whole repo. Pay particular attention to `_RevStack.backward` gradient ordering and to AMP interaction. Fix any confirmed findings, re-run the tests, then commit.

- [ ] **Step 6: Commit** (the notebooks are generated in Task 8, once the repo URL is known)

```bash
git add -A && git commit -m "feat: add Colab notebook generator

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Publish the repo to GitHub (needs your confirmation)

`gh` is not installed on this machine, so the GitHub repository is created in the browser.

- [ ] **Step 1 (you):** On github.com, create a **public**, empty repository named `reversible-llm` with no README or license. Send Claude the URL.
- [ ] **Step 2: Generate the notebooks with the real URL**

```bash
cd /c/adi-python/reversible-llm && .venv/Scripts/python -m revllm.notebooks https://github.com/<your-username>/reversible-llm.git
```
Save the printed badge lines for the README in Task 10.

- [ ] **Step 3: Commit and push.** Claude asks before pushing, because a push is outward-facing.

```bash
git add notebooks && git commit -m "feat: generate Colab notebooks

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git remote add origin https://github.com/<your-username>/reversible-llm.git && git push -u origin main
```

---

### Task 9: Run the experiments on Colab (you run these; Claude cannot drive Colab)

Estimated T4 time: about 15 min for the baseline, 20 min for the reversible run at the same batch, 10 min for the pilots, and 15–25 min for the max-batch run including the search. The total is roughly 1–1.5 h.

- [ ] **Step 1:** Open `01_baseline.ipynb` from the Colab badge. Set Runtime → T4 GPU → Run all. Check that `pytest` passes, including the GPU memory test, and that `01_baseline.json` appears in Drive.
- [ ] **Step 2:** Run `02_reversible_same_batch.ipynb`. It produces four pilot JSONs, `winner.json` and the full reversible run.
- [ ] **Step 3:** Run `03_reversible_max_batch.ipynb`.
- [ ] **Step 4:** Save each executed notebook **with outputs** back to the repo. Either use Colab's File → Save a copy in GitHub → `main` → `notebooks/<same name>`, or use File → Download .ipynb and copy it into `C:\adi-python\reversible-llm\notebooks\`.
- [ ] **Step 5:** Download `MyDrive/reversible-llm/results/*.json` and `batch_search_*.json` into `C:\adi-python\reversible-llm\results\`.

If a run diverges, OOMs or gets disconnected, bring Claude the error or cell output. Claude will use `superpowers:systematic-debugging` to find the cause. Common fixes are lower `h`, lower `lr`, or a smaller batch.

---

### Task 10: Analysis + README

**Files:**
- Execute: `notebooks/04_analysis.ipynb` (locally)
- Create: `README.md`, `results/plots/*.png`

- [ ] **Step 1: Execute the analysis notebook locally**

```bash
cd /c/adi-python/reversible-llm && git pull && .venv/Scripts/python -m jupyter nbconvert --to notebook --execute --inplace notebooks/04_analysis.ipynb
```
Expected: `results/plots/` contains the 4 PNGs and the notebook shows the table. Load the `dataviz` skill and check the plots against it.

- [ ] **Step 2: Write `README.md`** with exactly these sections, filled with numbers taken from the JSONs:
  1. **Title + TL;DR:** 3 bullets covering memory saved at the same batch (GB and %), speed cost (%), and max batch for baseline vs reversible (×).
  2. **Open in Colab:** the 4 badge lines from Task 8.
  3. **Setup:** GPU name, AMP dtype, dataset (TinyStories, GPT-2 BPE, 55M train / 1M val tokens), model table (d=256, 8 layers, 8 heads, ctx 512, 19.3M params, tied embeddings), optimizer (AdamW β=(0.9, 0.95), wd 0.1, cosine 1e-3→1e-4, warmup 200, clip 1.0), and the metric definitions from "Key design decisions" 4–5.
  4. **What reversibility is:** a short beginner-friendly explanation, the two update equations, and why backward only needs the final streams.
  5. **Variants tried (which worked):** the pilot table from `results_table` (the `02_pilot_*` rows plus reconstruction error), the `val_loss_pilots.png` plot, and a plain statement of the winner and why. Divergence, if any, is reported as a result, not hidden.
  6. **Results (the three required runs):** `results_table` for `01_*`, `02_reversible_*` and `03_*`, then `val_loss_full_runs.png`, `peak_memory.png` and `throughput.png`, plus baseline vs reversible max batch from `batch_search_*.json`.
  7. **Findings:**
     - (a) Memory vs compute trade-off: one extra forward pass per layer in backward.
     - (b) Why chunked loss was necessary: logits memory dominates at a 50k vocab.
     - (c) Equal-token budget at max batch means far fewer optimizer steps, so check whether the loss is worse and relate it to critical batch size.
     - (d) fp16 reconstruction error magnitude and whether it hurt training.
     - (e) Midpoint/leapfrog stability vs `h`.
     - (f) Anything surprising.
  8. **Reproduce:** the local test command and the Colab run order.
  9. **Repo layout:** the tree from this plan.

- [ ] **Step 3: Verify.** Invoke `superpowers:verification-before-completion`. Run `.venv/Scripts/python -m pytest tests -q` and confirm every number in the README matches the JSON it came from. A small script that prints each JSON's key metrics next to the README table is the easiest way to check.

- [ ] **Step 4: Review.** Invoke `code-review` on the final diff and fix any confirmed findings.

- [ ] **Step 5: Commit + push** (confirm with you first)

```bash
git add -A && git commit -m "docs: add results, plots and detailed README

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push
```

---

### Task 11: Public-link check and submission

- [ ] **Step 1:** Claude opens `https://github.com/<your-username>/reversible-llm/blob/main/README.md` in the built-in browser pane, which does not have your GitHub login. It confirms the README renders, the images load, the 4 notebooks are listed under `notebooks/` and show their outputs, and the Colab badges resolve.
- [ ] **Step 2 (you):** Open the same link in an incognito window yourself. The checkbox is your attestation.
- [ ] **Step 3 (you):** Paste the README link into the submission form with a caption such as "README: 20M GPT on 50M tokens, baseline vs reversible (Euler/midpoint) vs max-batch reversible, with Colab notebooks". Tick the box and click Submit. Claude does not submit forms on your behalf.

---

## Self-review notes

- **Spec coverage:** every assignment checkpoint maps to a task (see the table above). The "other findings" items are listed explicitly in Task 10, Step 2.7.
- **Consistency:** the names used across tasks are fixed: `train()`, `find_max_batch()`, `load_results()`, `results_table()`, `plot_all()`, `reconstruction_error()`, `GPT.embed()` and `GPT.head()`. The run names `01_baseline`, `02_pilot_<arch>_h<h>`, `02_reversible_<arch>_same_batch` and `03_reversible_<arch>_max_batch` are what `plot_all` uses to split pilots from full runs.
- **Known risks:**
  - Midpoint/leapfrog can be unstable on dissipative dynamics. The `h=0.25` pilot is the fallback.
  - A very large max batch leaves few optimizer steps, which is expected and reported as a finding.
  - Colab disconnects are covered because results are saved to Drive after each run.
