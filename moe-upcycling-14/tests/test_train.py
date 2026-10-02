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
