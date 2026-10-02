# Session 14 — Dense → MoE Upcycling: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a public GitHub folder whose `README.md` shows, with committed training logs and plots, that (1) a small dense ("Linear") GPT was trained from scratch and its loss fell, and (2) the same model was converted into a Mixture-of-Experts by *sparse upcycling* and **continued to train and reduce loss** — compared against a dense model trained for the same extra steps.

**Architecture:** A tiny character-level GPT whose feed-forward block is two `nn.Linear` layers. After dense pre-training, every FFN is replaced by an `MoE` layer holding `E=4` copies of that FFN plus a freshly initialised router (top-2 routing, Switch-style load-balancing loss). Because all experts are identical and the top-2 gates are renormalised to sum to 1, the converted model computes *exactly* the dense function at step 0 — a unit test pins this. Training then continues for both the MoE and a dense control from the same checkpoint with the same schedule, data order and evaluation batches.

**Tech Stack:** Python 3.12, PyTorch 2.x (the installed CPU build is enough — see the time budget), NumPy, Matplotlib, pytest. No `transformers`, no tokenizer library.

**Spec:** The assignment brief, reproduced verbatim below.

---

## Assignment Brief (the spec)

> Train a Linear model and convert that into an MoE! Your call on model size and data trained on, but must show they continue to train and reduce loss!

**Deliverable:** "GitHub README Link" — *The Repo MUST have training logs.* Public link, verified in an incognito window, with a short caption. 1000 points, resubmission allowed.

**Due: Sat Oct 3 2026, 07:00.** At the time of writing (02:12 IST, same day) that leaves **≈4 h 45 min**. The plan is sized to finish in ≈3 h with slack. Resubmission is allowed, so submit a working version first and improve after.

### How "Linear model" is interpreted

"Linear" is read as **dense**: an ordinary transformer whose FFN is `Linear → GELU → Linear` and runs for every token. That FFN is exactly the part an MoE replaces, so "convert it into an MoE" means *turn each dense FFN into a set of experts with a router*. The README states this interpretation in its first section, in one sentence, so a grader who meant something else can see the choice was deliberate.

---

## Global Constraints

- **Location:** `C:\adi-python\llm_finetune_assignments\moe-upcycling-14\` inside the existing repo `AditeyaAItronics/llm_finetune_assignments` (branch `main`). All paths below are relative to this folder unless stated.
- **Do not touch** the repo-root `README.md` — it has uncommitted edits by the user. Stage only files under `moe-upcycling-14/`.
- **Determinism:** every script calls `set_seed(1337)` before building anything. Evaluation uses a fixed generator (`EVAL_SEED = 999`) so every run is scored on *identical* batches.
- **Data:** Tiny Shakespeare (1,115,394 chars, 65-char vocab), committed at `data/tinyshakespeare.txt` so graders can run offline. 90/10 train/val split by position.
- **Model (fixed):** `d_model=256, n_layers=4, n_heads=4, d_ff=1024, seq_len=128, vocab=65` → ≈3.2 M params dense.
- **MoE (fixed):** `n_experts=4, top_k=2`, router `Linear(256, 4, bias=False)` init `N(0, 0.02)`, aux-loss coefficient `0.01`. Every FFN layer becomes an MoE layer.
- **Function preservation is mandatory:** the upcycled model's logits must match the dense model's to `atol=1e-5` before any MoE training. If this fails, nothing after it is valid.
- **Fair control:** dense-continue and MoE-continue start from the *same* `checkpoints/dense.pt`, with a *fresh* AdamW, the *same* LR schedule, the *same* training seed, the same step count and the same eval batches. The only difference is the FFN.
- **Training logs must be committed:** `logs/*.log` (human-readable) and `logs/*.jsonl` (machine-readable) for all three runs. `checkpoints/` is git-ignored.
- **Every number in `README.md` comes from `results/summary.json`** produced by `experiments/exp3_report.py`. No hand-typed numbers.
- **Plots:** matplotlib only, axis labels + title + legend on every figure, saved at 150 dpi.
- **Commits:** this repo's existing convention — short imperative subject, body optional, ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Tone:** formal and professional in the README.

---

## Background the executor needs

**Sparse upcycling** (Komatsuzaki et al., 2022) converts a trained dense model into an MoE by copying each dense FFN into every expert and adding a new router. It is a "warm start": the MoE begins at the dense model's loss instead of from random weights, then uses its extra parameters to keep improving.

**Why the conversion is exactly function-preserving here.** For a token `x`, the MoE output is `Σ_{e ∈ topk} g_e · FFN_e(x)` with `g` the top-k router probabilities renormalised to sum to 1. When every `FFN_e` is the same function `FFN`, this equals `FFN(x) · Σ g_e = FFN(x)`, whatever the router says. So the router can start random (which spreads tokens across experts from step 0) without changing the output. Adding noise to the expert copies would break this — **do not add noise**. Experts diverge on their own because each receives gradients only from the tokens routed to it.

**Load-balancing loss (Switch Transformer).** `aux = E · Σ_e f_e · P_e`, where `f_e` is the fraction of routing slots sent to expert `e` and `P_e` is the mean router probability for `e`. It is ≈1 when balanced and grows when one expert hogs tokens. Total loss = `CE + 0.01 · Σ_layers aux`. Without it, routers often collapse onto one or two experts.

**What the README must make visible.** (a) Dense pre-training loss falls. (b) At the conversion point the MoE's loss is identical to the dense model's (no spike). (c) After conversion, MoE loss keeps falling. (d) The dense control also keeps falling — so the README must compare MoE-continue against dense-continue, not against the frozen dense checkpoint, otherwise "MoE helped" cannot be separated from "more training helped". (e) Experts are actually used (routing fractions per layer over time).

**Cost honesty.** Top-2 of 4 experts means each token runs **2 FFNs**, so active FFN FLOPs per token are 2× the dense model, and total FFN parameters are 4×. The README states this; any MoE advantage is reported alongside it.

---

## Time budget (CPU, 16 threads, measured 0.57 s/step for the dense model at batch 32)

| Clock (IST) | Task | Wall time |
|---|---|---|
| 02:20–02:35 | Tasks 1–2: scaffold, data, dense model, tests | 15 min |
| 02:35–02:55 | Tasks 3–4: MoE layer, upcycling, tests | 20 min |
| 02:55–03:10 | Task 5: training loop + logging, tests | 15 min |
| 03:10–03:30 | Task 6: dense pre-train, 1500 steps | ≈15 min run |
| 03:30–04:05 | Task 7: MoE-continue (≈17 min) then dense-continue (≈10 min), 1000 steps each | ≈30 min run |
| 04:05–04:20 | Task 8: report script, plots, summary | 15 min |
| 04:20–04:50 | Task 9: README | 30 min |
| 04:50–05:05 | Task 10: commit, push, incognito check, submit | 15 min |

≈2 h buffer before 07:00. If a run is slower than expected, cut continuation to 600 steps (change one constant) rather than missing the deadline. While a run is going, write the next task's code.

---

## File Structure

```
moe-upcycling-14/
├── plan.md                    # this file
├── README.md                  # graded artifact (Task 9)
├── requirements.txt           # torch, numpy, matplotlib, pytest
├── .gitignore                 # checkpoints/, __pycache__/
├── data/tinyshakespeare.txt   # committed corpus
├── src/
│   ├── __init__.py
│   ├── seeding.py             # set_seed
│   ├── data.py                # CharData: load, split, batch, decode
│   ├── model.py               # GPTConfig, MLP, Attention, Block, GPT
│   ├── moe.py                 # MoE layer + upcycle(dense) -> moe model
│   └── train.py               # TrainConfig, lr_at, evaluate, train (logging)
├── experiments/
│   ├── __init__.py
│   ├── exp1_pretrain_dense.py # dense from scratch -> checkpoints/dense.pt
│   ├── exp2_continue.py       # --arch moe|dense, continue from dense.pt
│   └── exp3_report.py         # logs -> plots + summary.json/md + samples
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_moe.py
│   ├── test_upcycle.py
│   └── test_train.py
├── logs/                      # COMMITTED training logs (.log + .jsonl)
├── results/                   # COMMITTED plots + summary
└── checkpoints/               # git-ignored
```

All commands below run from `C:\adi-python\llm_finetune_assignments\moe-upcycling-14`. Tests import as `from src.model import GPT`; `pytest` finds `src` because the tests are run from this folder with `python -m pytest` (which puts the cwd on `sys.path`).

---

### Task 1: Scaffold, data, seeding

**Files:**
- Create: `requirements.txt`, `.gitignore`, `src/__init__.py`, `experiments/__init__.py`, `src/seeding.py`, `src/data.py`, `data/tinyshakespeare.txt`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `set_seed(seed: int) -> None`; `CharData(path=DATA_PATH, val_frac=0.1)` with attributes `vocab_size: int`, `train: LongTensor`, `val: LongTensor`, methods `batch(split: str, batch_size: int, seq_len: int, generator: torch.Generator) -> (x, y)` and `decode(ids: list[int]) -> str`.

- [ ] **Step 1: Create folders, static files, and fetch the corpus**

```bash
mkdir -p moe-upcycling-14/{src,experiments,tests,data,logs,results,checkpoints}
cd moe-upcycling-14
curl -sSL -o data/tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
wc -c data/tinyshakespeare.txt   # expect 1115394
touch src/__init__.py experiments/__init__.py
```

If `curl` resets (it has before on this machine), open the same URL in the built-in browser and save the text.

`requirements.txt`:
```
torch>=2.1
numpy>=1.24
matplotlib>=3.7
pytest>=7.4
```

`.gitignore`:
```
checkpoints/
__pycache__/
*.pyc
```

- [ ] **Step 2: Write the failing test** — `tests/test_data.py`

```python
import torch
from src.data import CharData


def test_vocab_and_split():
    d = CharData()
    assert d.vocab_size == 65
    total = d.train.numel() + d.val.numel()
    assert total == 1115394
    assert abs(d.val.numel() / total - 0.1) < 1e-3


def test_batch_shapes_and_shift():
    d = CharData()
    g = torch.Generator().manual_seed(0)
    x, y = d.batch("train", 8, 16, g)
    assert x.shape == y.shape == (8, 16)
    assert x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])  # y is x shifted by one


def test_batch_is_deterministic():
    d = CharData()
    a = d.batch("val", 4, 32, torch.Generator().manual_seed(5))
    b = d.batch("val", 4, 32, torch.Generator().manual_seed(5))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_decode_roundtrip():
    d = CharData()
    assert d.decode(d.train[:14].tolist()) == "First Citizen:"
```

- [ ] **Step 3: Run it — expect FAIL** (`ModuleNotFoundError: src.data`)

Run: `python -m pytest tests/test_data.py -v`

- [ ] **Step 4: Implement** `src/seeding.py`

```python
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

`src/data.py`:
```python
"""Character-level Tiny Shakespeare. Committed to the repo so runs are offline and exact."""
from pathlib import Path

import torch

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"


class CharData:
    def __init__(self, path: Path = DATA_PATH, val_frac: float = 0.1):
        text = Path(path).read_text(encoding="utf-8")
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(ids) * (1 - val_frac))
        self.train, self.val = ids[:n], ids[n:]
        self.vocab_size = len(self.chars)

    def batch(self, split: str, batch_size: int, seq_len: int, generator: torch.Generator):
        src = self.train if split == "train" else self.val
        ix = torch.randint(0, src.numel() - seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([src[i:i + seq_len] for i in ix])
        y = torch.stack([src[i + 1:i + 1 + seq_len] for i in ix])
        return x, y

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)
```

- [ ] **Step 5: Run — expect 4 PASS.** `python -m pytest tests/test_data.py -v`

- [ ] **Step 6: Commit**

```bash
git add moe-upcycling-14/requirements.txt moe-upcycling-14/.gitignore moe-upcycling-14/src moe-upcycling-14/experiments/__init__.py moe-upcycling-14/data moe-upcycling-14/tests/test_data.py moe-upcycling-14/plan.md
git commit -m "Add Session 14 scaffold: Tiny Shakespeare data loader and plan" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Dense ("Linear") GPT

**Files:**
- Create: `src/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `GPTConfig` dataclass (`vocab_size=65, seq_len=128, d_model=256, n_layers=4, n_heads=4, d_ff=1024`); `MLP(d_model, d_ff)` with submodules `fc1`, `fc2`; `Block(cfg)` with attributes `ln1, attn, ln2, ffn` (**`ffn` is the attribute Task 4 swaps**); `GPT(cfg)` with `.blocks: nn.ModuleList`, `forward(idx, targets=None) -> (logits, ce_loss_or_None)`, `aux_loss() -> Tensor`, `num_params() -> int`, `generate(idx, n_new, temperature=0.8, generator=None) -> LongTensor`.

- [ ] **Step 1: Write the failing test** — `tests/test_model.py`

```python
import math

import torch
from src.model import GPT, GPTConfig

TINY = GPTConfig(vocab_size=65, seq_len=16, d_model=32, n_layers=2, n_heads=2, d_ff=64)


def test_forward_shapes_and_initial_loss():
    torch.manual_seed(0)
    m = GPT(TINY)
    x = torch.randint(0, 65, (3, 16))
    logits, loss = m(x, x)
    assert logits.shape == (3, 16, 65)
    # freshly initialised model should be near uniform: ln(65) ~= 4.17
    assert abs(loss.item() - math.log(65)) < 0.5


def test_causality():
    torch.manual_seed(0)
    m = GPT(TINY).eval()
    x = torch.randint(0, 65, (1, 16))
    x2 = x.clone()
    x2[0, 10] = (x2[0, 10] + 1) % 65
    l1, _ = m(x)
    l2, _ = m(x2)
    assert torch.allclose(l1[0, :10], l2[0, :10], atol=1e-6)  # future token must not leak back


def test_dense_aux_loss_is_zero():
    m = GPT(TINY)
    assert m.aux_loss().item() == 0.0


def test_default_param_count():
    n = GPT(GPTConfig()).num_params()
    assert 3_000_000 < n < 3_500_000
```

- [ ] **Step 2: Run — expect FAIL.** `python -m pytest tests/test_model.py -v`

- [ ] **Step 3: Implement** `src/model.py`

```python
"""A small dense GPT. Its FFN (`Block.ffn`) is Linear -> GELU -> Linear: the part MoE replaces."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    seq_len: int = 128
    d_model: int = 256
    n_layers: int = 4
    n_heads: int = 4
    d_ff: int = 1024


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x):
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=2)
        q, k, v = (z.view(b, t, self.n_heads, d // self.n_heads).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(b, t, d))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = MLP(cfg.d_model, cfg.d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        t = idx.shape[1]
        h = self.tok(idx) + self.pos(torch.arange(t, device=idx.device))[None]
        for blk in self.blocks:
            h = blk(h)
        logits = self.head(self.ln_f(h))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

    def aux_loss(self) -> torch.Tensor:
        """Sum of MoE load-balancing losses from the last forward; 0 for a dense model."""
        total = torch.zeros(())
        for blk in self.blocks:
            if hasattr(blk.ffn, "aux_loss"):
                total = total + blk.ffn.aux_loss
        return total

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def generate(self, idx, n_new: int, temperature: float = 0.8, generator=None):
        for _ in range(n_new):
            logits, _ = self(idx[:, -self.cfg.seq_len:])
            probs = F.softmax(logits[:, -1] / temperature, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1, generator=generator)], dim=1)
        return idx
```

- [ ] **Step 4: Run — expect 4 PASS.** `python -m pytest tests/test_model.py -v`. If `test_default_param_count` fails, print `num_params()` and adjust the bounds to the real value ±10% (the exact count is ≈3.2 M; the test exists to catch a wrong config, not to be clever).

- [ ] **Step 5: Commit** — `git add moe-upcycling-14/src/model.py moe-upcycling-14/tests/test_model.py` then commit `"Add dense character-level GPT for Session 14"` with the co-author trailer.

---

### Task 3: MoE layer

**Files:**
- Create: `src/moe.py` (MoE class only in this task)
- Test: `tests/test_moe.py`

**Interfaces:**
- Consumes: `MLP` from `src.model`.
- Produces: `MoE(d_model, d_ff, n_experts=4, top_k=2)` with `.router: nn.Linear(d_model, n_experts, bias=False)`, `.experts: nn.ModuleList[MLP]`, `forward(x[B,T,D]) -> [B,T,D]`; after each forward sets `.aux_loss` (scalar tensor with grad) and `.last_frac` (detached float tensor `[n_experts]`, fraction of routing slots per expert, sums to 1).

- [ ] **Step 1: Write the failing test** — `tests/test_moe.py`

```python
import torch
from src.moe import MoE


def test_shapes_and_fractions():
    torch.manual_seed(0)
    moe = MoE(16, 32, n_experts=4, top_k=2)
    y = moe(torch.randn(2, 5, 16))
    assert y.shape == (2, 5, 16)
    assert moe.last_frac.shape == (4,)
    assert abs(moe.last_frac.sum().item() - 1.0) < 1e-6


def test_matches_naive_reference():
    """Vectorised dispatch must equal a per-token loop."""
    torch.manual_seed(0)
    moe = MoE(8, 16, n_experts=4, top_k=2)
    x = torch.randn(1, 6, 8)
    y = moe(x)[0]
    h = x[0]
    probs = moe.router(h).softmax(-1)
    for i in range(6):
        v, idx = probs[i].topk(2)
        g = v / v.sum()
        ref = sum(g[j] * moe.experts[idx[j]](h[i]) for j in range(2))
        assert torch.allclose(y[i], ref, atol=1e-6)


def test_router_and_experts_get_gradients():
    torch.manual_seed(0)
    moe = MoE(16, 32, n_experts=4, top_k=2)
    y = moe(torch.randn(4, 8, 16))
    (y.pow(2).mean() + moe.aux_loss).backward()
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0
    used = [e for e in moe.experts if e.fc1.weight.grad is not None]
    assert len(used) >= 2


def test_aux_loss_is_one_when_perfectly_balanced():
    moe = MoE(4, 8, n_experts=4, top_k=1)
    with torch.no_grad():
        moe.router.weight.copy_(torch.eye(4) * 50)  # token i -> expert i, prob ~1
    moe(torch.eye(4).view(1, 4, 4))
    assert abs(moe.aux_loss.item() - 1.0) < 1e-3
```

- [ ] **Step 2: Run — expect FAIL.** `python -m pytest tests/test_moe.py -v`

- [ ] **Step 3: Implement** `src/moe.py`

```python
"""Top-k Mixture-of-Experts FFN and sparse upcycling of a dense GPT."""
import torch
import torch.nn as nn

from src.model import MLP


class MoE(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([MLP(d_model, d_ff) for _ in range(n_experts)])
        self.aux_loss = torch.zeros(())
        self.last_frac = torch.full((n_experts,), 1.0 / n_experts)

    def forward(self, x):
        b, t, d = x.shape
        h = x.reshape(-1, d)
        probs = self.router(h).softmax(dim=-1)                 # [N, E]
        top_p, top_i = probs.topk(self.top_k, dim=-1)          # [N, k]
        gates = top_p / top_p.sum(dim=-1, keepdim=True)        # renormalise: sums to 1
        out = torch.zeros_like(h)
        for e, expert in enumerate(self.experts):
            tok, slot = (top_i == e).nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            out.index_add_(0, tok, gates[tok, slot, None] * expert(h[tok]))
        # Switch Transformer load-balancing loss: E * sum_e f_e * P_e (=1 when balanced)
        counts = torch.bincount(top_i.flatten(), minlength=self.n_experts).float()
        frac = counts / counts.sum()
        self.aux_loss = self.n_experts * (frac * probs.mean(dim=0)).sum()
        self.last_frac = frac.detach()
        return out.reshape(b, t, d)
```

- [ ] **Step 4: Run — expect 4 PASS.**

- [ ] **Step 5: Commit** — `"Add top-k MoE layer with Switch load-balancing loss"`.

---

### Task 4: Sparse upcycling (dense → MoE conversion)

**Files:**
- Modify: `src/moe.py` (append `upcycle`)
- Test: `tests/test_upcycle.py`

**Interfaces:**
- Consumes: `GPT`, `GPTConfig` from `src.model`; `MoE` from Task 3.
- Produces: `upcycle(dense: GPT, n_experts: int = 4, top_k: int = 2, router_std: float = 0.02, seed: int = 0) -> GPT` — returns a **new** model (dense is untouched) whose every `blocks[i].ffn` is an `MoE` with all experts copied from the dense FFN.

- [ ] **Step 1: Write the failing test** — `tests/test_upcycle.py`

```python
import torch
from src.model import GPT, GPTConfig, MLP
from src.moe import MoE, upcycle

TINY = GPTConfig(vocab_size=65, seq_len=16, d_model=32, n_layers=2, n_heads=2, d_ff=64)


def _trained_like_dense():
    torch.manual_seed(0)
    m = GPT(TINY)
    with torch.no_grad():  # make FFN weights non-trivial, like a trained model
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    return m.eval()


def test_upcycle_is_function_preserving():
    dense = _trained_like_dense()
    moe = upcycle(dense).eval()
    x = torch.randint(0, 65, (4, 16))
    ld, _ = dense(x)
    lm, _ = moe(x)
    assert torch.allclose(ld, lm, atol=1e-5)


def test_structure_and_param_count():
    dense = _trained_like_dense()
    moe = upcycle(dense, n_experts=4)
    assert all(isinstance(b.ffn, MoE) for b in moe.blocks)
    assert all(isinstance(b.ffn, MLP) for b in dense.blocks)  # original untouched
    ffn = sum(p.numel() for p in dense.blocks[0].ffn.parameters())
    router = TINY.d_model * 4
    expected = dense.num_params() + TINY.n_layers * (3 * ffn + router)
    assert moe.num_params() == expected


def test_experts_are_independent_copies():
    dense = _trained_like_dense()
    moe = upcycle(dense)
    e0, e1 = moe.blocks[0].ffn.experts[0], moe.blocks[0].ffn.experts[1]
    assert torch.equal(e0.fc1.weight, e1.fc1.weight)
    with torch.no_grad():
        e0.fc1.weight.add_(1.0)
    assert not torch.equal(e0.fc1.weight, e1.fc1.weight)
    assert not torch.equal(e0.fc1.weight, dense.blocks[0].ffn.fc1.weight)


def test_router_spreads_tokens_at_init():
    dense = _trained_like_dense()
    moe = upcycle(dense)
    moe(torch.randint(0, 65, (8, 16)))
    frac = moe.blocks[0].ffn.last_frac
    assert (frac > 0).sum() >= 3  # random router must not send everything to one expert
```

- [ ] **Step 2: Run — expect FAIL** (`ImportError: upcycle`).

- [ ] **Step 3: Implement** — append to `src/moe.py`:

```python
import copy

from src.model import GPT


def upcycle(dense: GPT, n_experts: int = 4, top_k: int = 2,
            router_std: float = 0.02, seed: int = 0) -> GPT:
    """Sparse upcycling: copy each dense FFN into every expert, add a fresh router.

    Identical experts + gates that sum to 1 => output equals the dense FFN exactly,
    whatever the router picks. So the router is random (spreads load from step 0)
    and no noise is added to the experts (that would break function preservation).
    """
    model = copy.deepcopy(dense)
    g = torch.Generator().manual_seed(seed)
    for blk in model.blocks:
        dense_ffn = blk.ffn
        moe = MoE(model.cfg.d_model, model.cfg.d_ff, n_experts, top_k)
        for expert in moe.experts:
            expert.load_state_dict(dense_ffn.state_dict())
        with torch.no_grad():
            moe.router.weight.copy_(torch.randn(moe.router.weight.shape, generator=g) * router_std)
        blk.ffn = moe
    return model
```

Move the two new imports (`copy`, `GPT`) to the top of the file with the others.

- [ ] **Step 4: Run all tests — expect PASS.** `python -m pytest -v`

- [ ] **Step 5: Commit** — `"Add sparse upcycling from dense GPT to MoE (function-preserving)"`.

---

### Task 5: Training loop with logs

**Files:**
- Create: `src/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `CharData`, `GPT`, `set_seed`.
- Produces:
  - `TrainConfig(run_name: str, steps: int, batch_size=32, lr=1e-3, min_lr=1e-4, warmup=100, weight_decay=0.1, aux_coef=0.01, grad_clip=1.0, eval_every=100, eval_batches=20, log_every=10, seed=1337, step_offset=0)`
  - `lr_at(step: int, cfg: TrainConfig) -> float` (linear warmup to `lr`, cosine to `min_lr` at `steps`)
  - `evaluate(model, data, cfg) -> {"train": float, "val": float}` (fixed `EVAL_SEED=999` batches)
  - `train(model, data, cfg, log_dir: Path) -> list[dict]` — writes `log_dir/{run_name}.jsonl` and `log_dir/{run_name}.log`; returns the list of logged rows.
- JSONL row keys (Task 8 reads these): `step`, `global_step`, `lr`, `train_loss` (CE only), `aux_loss`, `tokens_seen`, `elapsed_s`; on eval rows also `val_loss`, `eval_train_loss`; on MoE runs also `expert_frac` (list per layer of list per expert).

- [ ] **Step 1: Write the failing test** — `tests/test_train.py`

```python
import json

import torch
from src.data import CharData
from src.model import GPT, GPTConfig
from src.moe import upcycle
from src.train import TrainConfig, evaluate, lr_at, train

TINY = GPTConfig(vocab_size=65, seq_len=32, d_model=32, n_layers=2, n_heads=2, d_ff=64)


def test_lr_schedule():
    c = TrainConfig(run_name="x", steps=100, lr=1e-3, min_lr=1e-4, warmup=10)
    assert abs(lr_at(0, c) - 1e-4) < 1e-12  # first warmup step = lr / warmup
    assert abs(lr_at(10, c) - 1e-3) < 1e-12
    assert abs(lr_at(100, c) - 1e-4) < 1e-12
    assert all(lr_at(s, c) >= lr_at(s + 1, c) for s in range(10, 100))


def test_evaluate_is_deterministic():
    d = CharData()
    torch.manual_seed(0)
    m = GPT(TINY)
    c = TrainConfig(run_name="x", steps=1, batch_size=4, eval_batches=2)
    assert evaluate(m, d, c) == evaluate(m, d, c)


def test_short_run_reduces_loss_and_logs(tmp_path):
    d = CharData()
    torch.manual_seed(0)
    m = GPT(TINY)
    c = TrainConfig(run_name="t", steps=60, batch_size=8, lr=3e-3, warmup=5,
                    eval_every=30, eval_batches=2, log_every=10)
    rows = train(m, d, c, tmp_path)
    evals = [r for r in rows if "val_loss" in r]
    assert evals[0]["step"] == 0 and evals[-1]["step"] == 60
    assert evals[-1]["val_loss"] < evals[0]["val_loss"] - 0.3
    lines = (tmp_path / "t.jsonl").read_text().strip().splitlines()
    assert len(lines) == len(rows) and json.loads(lines[0])["step"] == 0
    assert (tmp_path / "t.log").read_text().count("step") >= 7


def test_moe_run_logs_expert_fractions(tmp_path):
    d = CharData()
    torch.manual_seed(0)
    m = upcycle(GPT(TINY))
    c = TrainConfig(run_name="m", steps=10, batch_size=4, eval_every=10, eval_batches=1,
                    log_every=5, step_offset=500)
    rows = train(m, d, c, tmp_path)
    assert rows[-1]["global_step"] == 510
    assert len(rows[-1]["expert_frac"]) == 2 and len(rows[-1]["expert_frac"][0]) == 4
    assert rows[-1]["aux_loss"] > 0
```

- [ ] **Step 2: Run — expect FAIL.**

- [ ] **Step 3: Implement** `src/train.py`

```python
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
```

- [ ] **Step 4: Run all tests — expect PASS** (`python -m pytest -v`, ≈20–40 s on CPU). If `test_short_run_reduces_loss_and_logs` misses the 0.3 margin, raise `steps` to 100 in the test, not the margin.

- [ ] **Step 5: Commit** — `"Add training loop with JSONL and text training logs"`.

---

### Task 6: Dense pre-training run

**Files:**
- Create: `experiments/exp1_pretrain_dense.py`
- Produces on disk: `checkpoints/dense.pt`, `logs/dense_pretrain.log`, `logs/dense_pretrain.jsonl`, `results/environment.txt`

**Interfaces:**
- Consumes: `CharData`, `GPT`, `GPTConfig`, `TrainConfig`, `train`, `set_seed`.
- Produces: `checkpoints/dense.pt` = `{"model": state_dict, "cfg": asdict(GPTConfig), "steps": DENSE_STEPS}`; constant `DENSE_STEPS = 1500` importable from this module.

- [ ] **Step 1: Write the script**

```python
"""Stage 1: train the dense ("Linear") GPT from scratch."""
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import CharData  # noqa: E402
from src.model import GPT, GPTConfig  # noqa: E402
from src.seeding import set_seed  # noqa: E402
from src.train import TrainConfig, train  # noqa: E402

DENSE_STEPS = 1500


def main():
    torch.set_num_threads(16)
    set_seed(1337)
    data = CharData()
    cfg = GPTConfig(vocab_size=data.vocab_size)
    model = GPT(cfg)
    tcfg = TrainConfig(run_name="dense_pretrain", steps=DENSE_STEPS, lr=1e-3, min_lr=1e-4, warmup=100)
    train(model, data, tcfg, ROOT / "logs")
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "steps": DENSE_STEPS},
               ROOT / "checkpoints" / "dense.pt")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "environment.txt").write_text(
        f"python {platform.python_version()}\ntorch {torch.__version__}\n"
        f"device cpu, threads {torch.get_num_threads()}\nplatform {platform.platform()}\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test** — temporarily run with `DENSE_STEPS=20` by editing the constant, confirm `logs/dense_pretrain.log` has rows and `checkpoints/dense.pt` exists, then set it back to 1500.

- [ ] **Step 3: Real run (≈15 min), in the background**

Run: `python experiments/exp1_pretrain_dense.py`
Expected: val loss ≈4.17 at step 0, falling below ≈1.9 by step 1500 (typical for this size on Tiny Shakespeare; exact value does not matter — monotone decrease does). Watch `logs/dense_pretrain.log`.

**While it runs, start Task 7's script and Task 8's report script.**

- [ ] **Step 4: Sanity checks** — `tail -3 logs/dense_pretrain.log`; the last eval val loss must be well under the step-0 value. If val loss rises in the last 500 steps (overfitting), note it for the README; do not retune unless it is severe.

- [ ] **Step 5: Commit** script + logs + environment (not the checkpoint): `"Run dense pre-training (1500 steps) and add training logs"`.

---

### Task 7: Convert and continue — MoE vs dense control

**Files:**
- Create: `experiments/exp2_continue.py`
- Produces on disk: `logs/moe_continue.{log,jsonl}`, `logs/dense_continue.{log,jsonl}`, `logs/conversion_check.json`, `checkpoints/{moe,dense}_continue.pt`

**Interfaces:**
- Consumes: `checkpoints/dense.pt`, `upcycle`, `train`, `evaluate`, `DENSE_STEPS`.
- Produces: `logs/conversion_check.json` = `{"dense_val": float, "moe_val": float, "abs_diff": float, "max_logit_diff": float, "dense_params": int, "moe_params": int, "active_params_per_token": int}`; constant `CONTINUE_STEPS = 1000`.

- [ ] **Step 1: Write the script**

```python
"""Stage 2: continue training from the dense checkpoint, as MoE (upcycled) or as dense (control)."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.exp1_pretrain_dense import DENSE_STEPS  # noqa: E402
from src.data import CharData  # noqa: E402
from src.model import GPT, GPTConfig  # noqa: E402
from src.moe import upcycle  # noqa: E402
from src.seeding import set_seed  # noqa: E402
from src.train import TrainConfig, evaluate, train  # noqa: E402

CONTINUE_STEPS = 1000
N_EXPERTS, TOP_K = 4, 2


def load_dense() -> GPT:
    ck = torch.load(ROOT / "checkpoints" / "dense.pt", map_location="cpu")
    model = GPT(GPTConfig(**ck["cfg"]))
    model.load_state_dict(ck["model"])
    return model


def conversion_check(dense: GPT, moe: GPT, data: CharData, tcfg: TrainConfig) -> dict:
    dense.eval(); moe.eval()
    x, _ = data.batch("val", 8, dense.cfg.seq_len, torch.Generator().manual_seed(0))
    with torch.no_grad():
        max_diff = (dense(x)[0] - moe(x)[0]).abs().max().item()
    dv, mv = evaluate(dense, data, tcfg)["val"], evaluate(moe, data, tcfg)["val"]
    ffn = sum(p.numel() for p in dense.blocks[0].ffn.parameters())
    n_layers = dense.cfg.n_layers
    active = moe.num_params() - n_layers * (N_EXPERTS - TOP_K) * ffn
    out = {"dense_val": dv, "moe_val": mv, "abs_diff": abs(dv - mv), "max_logit_diff": max_diff,
           "dense_params": dense.num_params(), "moe_params": moe.num_params(),
           "active_params_per_token": active}
    assert max_diff < 1e-4, f"upcycling is not function-preserving: {max_diff}"
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["moe", "dense"], required=True)
    args = p.parse_args()
    torch.set_num_threads(16)
    set_seed(1337)
    data = CharData()
    model = load_dense()
    tcfg = TrainConfig(run_name=f"{args.arch}_continue", steps=CONTINUE_STEPS,
                       lr=5e-4, min_lr=5e-5, warmup=50, step_offset=DENSE_STEPS)
    if args.arch == "moe":
        moe = upcycle(model, n_experts=N_EXPERTS, top_k=TOP_K)
        check = conversion_check(model, moe, data, tcfg)
        (ROOT / "logs" / "conversion_check.json").write_text(json.dumps(check, indent=2))
        print("conversion check:", check)
        model = moe
    train(model, data, tcfg, ROOT / "logs")
    torch.save({"model": model.state_dict(), "arch": args.arch},
               ROOT / "checkpoints" / f"{args.arch}_continue.pt")


if __name__ == "__main__":
    main()
```

Why these settings: a fresh optimizer for both arms (the dense Adam state has no natural mapping onto four experts plus a router, and giving it to only one arm would be unfair); a short 50-step re-warmup so the new router does not receive full-size updates on its first steps; peak LR halved to 5e-4 because the model is already trained.

- [ ] **Step 2: Smoke test** — set `CONTINUE_STEPS=10`, run `python experiments/exp2_continue.py --arch moe`; confirm `logs/conversion_check.json` shows `abs_diff` ≈ 0 and `max_logit_diff < 1e-4`, and the `.log` shows expert fractions. Restore 1000.

- [ ] **Step 3: Real runs, one after the other** (sharing 16 threads slows both):

```bash
python experiments/exp2_continue.py --arch moe     # ~17 min
python experiments/exp2_continue.py --arch dense   # ~10 min
```

Expected: both step-0 val losses equal the dense pre-train final val loss (same eval batches); both fall afterwards. MoE step-0 `aux_loss` ≈ 4 layers × ~1.0.

- [ ] **Step 4: Sanity checks**
  - `grep "step     0" logs/*_continue.log` — the two val numbers match each other and the dense final eval to 4 decimals.
  - Expert fractions: no expert in any layer stuck at 0.00 for the last 500 steps. If one is, note it as expert collapse and (only if time allows) rerun MoE with `aux_coef=0.05`.

- [ ] **Step 5: Commit** script + all logs: `"Upcycle dense GPT to 4-expert MoE and continue training against a dense control"`.

---

### Task 8: Report — plots, summary, samples

**Files:**
- Create: `experiments/exp3_report.py`
- Produces: `results/loss_full.png`, `results/loss_after_conversion.png`, `results/expert_usage.png`, `results/aux_loss.png`, `results/summary.json`, `results/summary.md`, `results/samples.txt`

**Interfaces:**
- Consumes: the three JSONL logs, `logs/conversion_check.json`, the three checkpoints.
- Produces: `summary.json` keys — `dense_pretrain: {start_val, end_val, steps}`, `conversion: <conversion_check.json>`, `moe_continue: {start_val, end_val, min_val, drop}`, `dense_continue: {same keys}`, `moe_minus_dense_end_val`, `train_val_gap: {moe, dense}`, `wall_clock_s: {dense_pretrain, moe_continue, dense_continue}`, `final_expert_frac`.

- [ ] **Step 1: Write the script**

```python
"""Turn the committed logs into plots and the numbers quoted in README.md."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data import CharData  # noqa: E402
from src.model import GPT, GPTConfig  # noqa: E402
from src.moe import upcycle  # noqa: E402

LOGS, RES = ROOT / "logs", ROOT / "results"


def load(name):
    return [json.loads(l) for l in (LOGS / f"{name}.jsonl").read_text().splitlines()]


def evals(rows):
    return [r for r in rows if "val_loss" in r]


def stats(rows):
    e = evals(rows)
    return {"start_val": e[0]["val_loss"], "end_val": e[-1]["val_loss"],
            "min_val": min(r["val_loss"] for r in e), "drop": e[0]["val_loss"] - e[-1]["val_loss"],
            "end_eval_train": e[-1]["eval_train_loss"], "steps": e[-1]["step"],
            "wall_s": rows[-1]["elapsed_s"]}


def main():
    RES.mkdir(exist_ok=True)
    dp, mc, dc = load("dense_pretrain"), load("moe_continue"), load("dense_continue")
    conv = json.loads((LOGS / "conversion_check.json").read_text())
    switch = mc[0]["global_step"]

    # 1. full timeline
    fig, ax = plt.subplots(figsize=(9, 5))
    for rows, lab, c in [(dp, "dense pre-train", "tab:gray"), (dc, "dense continue (control)", "tab:blue"),
                         (mc, "MoE continue (upcycled)", "tab:red")]:
        ax.plot([r["global_step"] for r in rows], [r["train_loss"] for r in rows], c=c, alpha=0.25, lw=0.8)
        e = evals(rows)
        ax.plot([r["global_step"] for r in e], [r["val_loss"] for r in e], c=c, marker="o", ms=3, label=f"{lab} (val)")
    ax.axvline(switch, ls="--", c="k", lw=1, label="dense → MoE conversion")
    ax.set(xlabel="global step", ylabel="cross-entropy loss (nats/char)",
           title="Dense pre-training, then upcycled MoE vs dense continuation")
    ax.legend(); fig.tight_layout(); fig.savefig(RES / "loss_full.png", dpi=150); plt.close(fig)

    # 2. zoom after conversion
    fig, ax = plt.subplots(figsize=(9, 5))
    for rows, lab, c in [(dc, "dense continue", "tab:blue"), (mc, "MoE continue", "tab:red")]:
        e = evals(rows)
        ax.plot([r["global_step"] for r in e], [r["val_loss"] for r in e], c=c, marker="o", label=f"{lab} val")
        ax.plot([r["global_step"] for r in e], [r["eval_train_loss"] for r in e], c=c, ls=":", label=f"{lab} train (eval)")
    ax.set(xlabel="global step", ylabel="loss", title="After conversion: both keep reducing loss")
    ax.legend(); fig.tight_layout(); fig.savefig(RES / "loss_after_conversion.png", dpi=150); plt.close(fig)

    # 3. expert usage per layer over time
    n_layers = len(mc[0]["expert_frac"])
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 3.5), sharey=True)
    steps = [r["global_step"] for r in mc]
    for li, ax in enumerate(axes):
        for e in range(len(mc[0]["expert_frac"][li])):
            ax.plot(steps, [r["expert_frac"][li][e] for r in mc], label=f"expert {e}")
        ax.axhline(1 / len(mc[0]["expert_frac"][li]), c="k", ls="--", lw=0.8)
        ax.set(title=f"layer {li}", xlabel="global step")
    axes[0].set_ylabel("fraction of routing slots"); axes[-1].legend(fontsize=7)
    fig.suptitle("Expert utilisation (dashed = perfectly balanced)")
    fig.tight_layout(); fig.savefig(RES / "expert_usage.png", dpi=150); plt.close(fig)

    # 4. aux loss
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, [r["aux_loss"] for r in mc], c="tab:red", label="sum over layers")
    ax.axhline(n_layers, c="k", ls="--", lw=0.8, label=f"balanced = {n_layers}")
    ax.set(xlabel="global step", ylabel="load-balancing loss", title="Router load-balancing loss")
    ax.legend(); fig.tight_layout(); fig.savefig(RES / "aux_loss.png", dpi=150); plt.close(fig)

    s_mc, s_dc, s_dp = stats(mc), stats(dc), stats(dp)
    summary = {
        "dense_pretrain": s_dp, "conversion": conv, "moe_continue": s_mc, "dense_continue": s_dc,
        "moe_minus_dense_end_val": s_mc["end_val"] - s_dc["end_val"],
        "train_val_gap": {"moe": s_mc["end_val"] - s_mc["end_eval_train"],
                          "dense": s_dc["end_val"] - s_dc["end_eval_train"]},
        "final_expert_frac": mc[-1]["expert_frac"],
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2))

    md = ["| Run | Steps | Val loss start | Val loss end | Drop | Wall time |", "|---|---:|---:|---:|---:|---:|"]
    for name, s in [("Dense pre-train", s_dp), ("Dense continue (control)", s_dc), ("MoE continue (upcycled)", s_mc)]:
        md.append(f"| {name} | {s['steps']} | {s['start_val']:.4f} | {s['end_val']:.4f} | {s['drop']:.4f} | {s['wall_s']/60:.1f} min |")
    (RES / "summary.md").write_text("\n".join(md) + "\n")

    # 5. text samples from the three checkpoints (same seed, same prompt)
    data = CharData()
    prompt = torch.tensor([[data.stoi[c] for c in "ROMEO:"]])
    out = []
    for tag, path in [("dense @1500", "dense.pt"), ("dense @2500", "dense_continue.pt"), ("MoE @2500", "moe_continue.pt")]:
        ck = torch.load(ROOT / "checkpoints" / path, map_location="cpu")
        model = GPT(GPTConfig(vocab_size=data.vocab_size))
        if path == "moe_continue.pt":
            model = upcycle(model)
        model.load_state_dict(ck["model"]); model.eval()
        ids = model.generate(prompt, 300, generator=torch.Generator().manual_seed(0))[0].tolist()
        out.append(f"===== {tag} =====\n{data.decode(ids)}\n")
    (RES / "samples.txt").write_text("\n".join(out), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run** `python experiments/exp3_report.py`. Open each PNG and check: conversion line visible, no loss spike at it, both continuation curves going down, legend not covering data.

- [ ] **Step 3: Decide the headline honestly** using `summary.json`:
  - Required claim (assignment): `moe_continue.drop > 0` and `dense_continue.drop > 0` → "both continue to train and reduce loss". ✅ is the expected outcome.
  - Comparative claim: if `|moe_minus_dense_end_val|` is below ≈0.01 (a single run has no error bar), write "no clear difference at this scale" rather than claiming a win. If MoE is lower by more, state it together with its 2× active FFN compute and 4× FFN parameters.
  - If MoE's `train_val_gap` is larger, say the extra capacity is partly going into memorising a 1 MB corpus.

- [ ] **Step 4: Commit** script + `results/`: `"Add report script, loss and expert-usage plots, and summary"`.

---

### Task 9: README (the graded artifact)

**Files:**
- Create: `README.md` (≈1,200–1,800 words, 4 figures)

- [ ] **Step 1: Write it with these sections, in order** (all numbers copied from `results/summary.json` / `summary.md` / `logs/conversion_check.json`):

1. **Title + one-paragraph summary** — "A 3.2 M-parameter dense character-level GPT was trained on Tiny Shakespeare, converted into a 4-expert top-2 Mixture-of-Experts by sparse upcycling, and trained further. Both the MoE and a dense control continued to reduce loss…" with the three headline numbers.
2. **Interpretation of the task** — "Linear model" = dense model whose FFN is two linear layers; what "convert to MoE" means here.
3. **Setup table** — data (chars, vocab, split), model config, MoE config, optimizer, schedules for both stages, batch × context = tokens/step, total tokens, hardware (`results/environment.txt`).
4. **How the conversion works** — the identical-experts argument in 4 lines, the router init, the load-balancing loss formula, and the measured `max_logit_diff` / `abs_diff` from `conversion_check.json` as proof it is exact.
5. **Results** — `summary.md` table; `loss_full.png`; `loss_after_conversion.png`; one paragraph per figure stating what to look at.
6. **Are the experts actually used?** — `expert_usage.png`, `aux_loss.png`, final fractions per layer.
7. **Parameters and compute** — dense params, MoE total params, active params per token, measured s/step for each arm (from wall time), and the statement that top-2 doubles active FFN FLOPs.
8. **Samples** — 3 short excerpts (≤ 6 lines each) from `results/samples.txt`.
9. **Limitations** — single seed; small corpus (overfitting risk); character-level; CPU-scale model; loop-based expert dispatch (not a fused kernel).
10. **Training logs** — explicit list with links: `logs/dense_pretrain.log`, `logs/moe_continue.log`, `logs/dense_continue.log` (+ `.jsonl`), and a 6-line excerpt from `moe_continue.log` showing step 0 and the last steps. **This section satisfies "The Repo MUST have training logs" — make it impossible to miss; also link it from the summary paragraph.**
11. **Reproduce** — exact commands:
    ```bash
    pip install -r requirements.txt
    python -m pytest -v
    python experiments/exp1_pretrain_dense.py
    python experiments/exp2_continue.py --arch moe
    python experiments/exp2_continue.py --arch dense
    python experiments/exp3_report.py
    ```
12. **File map** — one line per file.

- [ ] **Step 2: Self-check the README**
  - Every number appears in `results/` or `logs/` (search for each one).
  - All image links are relative (`results/loss_full.png`) and render on GitHub.
  - Formal tone; no casual phrasing.

- [ ] **Step 3: Commit** — `"Add Session 14 README: dense to MoE upcycling results and training logs"`.

---

### Task 10: Publish and submit

- [ ] **Step 1: Final checks**

```bash
python -m pytest -v                                   # all green
git status --short                                    # only intended files; root README.md stays unstaged
git check-ignore checkpoints/dense.pt                 # must print the path (ignored)
ls logs/                                              # 3 × .log, 3 × .jsonl, conversion_check.json
```

- [ ] **Step 2: Push** — `git push origin main` (confirm with the user before pushing).

- [ ] **Step 3: Verify publicly** — open `https://github.com/AditeyaAItronics/llm_finetune_assignments/tree/main/moe-upcycling-14` in a private window: README renders, all 4 images load, `logs/` folder is visible and the `.log` files open.

- [ ] **Step 4: Submit** (user does this in the portal)
  - URL: `https://github.com/AditeyaAItronics/llm_finetune_assignments/blob/main/moe-upcycling-14/README.md`
  - Caption: `Session 14: dense GPT upcycled to a 4-expert MoE — README with loss curves and training logs (logs/)`
  - Tick "I tested this link in an incognito window".

---

## Optional improvements (only after submitting; resubmission is allowed)

1. **Second seed for both continuation arms** (`--seed 2024`) → a noise floor for the MoE-vs-dense gap. ~30 min.
2. **Compute-matched comparison:** top-1 routing (same active FFN FLOPs as dense) as a third arm.
3. **MoE from scratch** for 2500 steps, to show what upcycling buys over random init at equal steps.
4. **GPU:** the laptop has an RTX 3050 6 GB but the installed torch is CPU-only; `pip install torch --index-url https://download.pytorch.org/whl/cu121` (≈2.5 GB) would allow a larger model / TinyStories.

---

## Self-review against the spec

| Spec requirement | Where it is met |
|---|---|
| Train a Linear (dense) model | Task 6, `logs/dense_pretrain.*`, `loss_full.png` |
| Convert it into an MoE | Task 4 (`upcycle`, exactness test), Task 7 (`conversion_check.json`) |
| Your call on size and data | Global Constraints + README §3 justify 3.2 M params / Tiny Shakespeare |
| Show they continue to train and reduce loss | Task 7 both arms, Task 8 `summary.json` drops, `loss_after_conversion.png` |
| Repo MUST have training logs | `logs/` committed (Tasks 6–7), README §10, `.gitignore` excludes only checkpoints |
| Public GitHub README link | Task 10 incognito check |
