import json

import numpy as np
import pytest
import torch

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
    for key in ("final_train_loss_window_tokens", "scaler_skipped_steps", "final_loss_scale",
                "recon_error_rel", "final_grad_cos", "final_grad_rel_err"):
        assert key in r
    assert r["final_train_loss_window_tokens"] > 0
    if arch != "baseline":
        assert r["recon_error"] < (1e-2 if torch.cuda.is_available() else 1e-3)
        assert r["final_grad_cos"] > 0.99
    else:
        assert r["final_grad_cos"] is None
        assert r["history"]["grad_cos"] == []
