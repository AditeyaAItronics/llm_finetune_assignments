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
    (tmp_path / "batch_search_baseline.json").write_text(
        json.dumps({"arch": "baseline", "max_batch": 100, "recommended": 88, "first_oom": 101}))
    rs = load_results(str(tmp_path))
    assert [r["run_name"] for r in rs] == ["01_baseline", "02_pilot_midpoint_h0.5"]  # batch_search_*.json ignored
    table = results_table(rs)
    assert "| 01_baseline | baseline |" in table
    assert "diverged" in table and "1.500" in table


def test_plot_all_writes_four_pngs(tmp_path):
    paths = plot_all([fake("01_baseline", "baseline", 64, 1.5), fake("02_pilot_euler_h1.0", "euler", 64, 1.6)], str(tmp_path))
    assert len(paths) == 4 and all(os.path.getsize(p) > 0 for p in paths)
