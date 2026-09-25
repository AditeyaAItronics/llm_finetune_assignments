import glob
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


COLORS = {"baseline": "#2a78d6", "euler": "#eb6834", "midpoint": "#1baf7a"}


def load_results(results_dir: str) -> list[dict]:
    rs = []
    for p in glob.glob(os.path.join(results_dir, "*.json")):
        with open(p) as f:
            r = json.load(f)
        if "run_name" not in r:  # e.g. batch_search_*.json, not a training run
            continue
        rs.append(r)
    return sorted(rs, key=lambda r: r["run_name"])


def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "—"
    return f"{v:,.{nd}f}"


def results_table(results: list[dict]) -> str:
    rows = ["| Run | Arch | h | Batch | Tokens | Final train loss | Final val loss | Tokens/s | Peak mem (GB) "
            "| Grad cos | Wall (min) | Status |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        h = "—" if r["arch"] == "baseline" else r["h"]
        rows.append(
            f"| {r['run_name']} | {r['arch']} | {h} | {r['batch_size']} | {r['tokens_seen'] / 1e6:.1f}M "
            f"| {_fmt(r['final_train_loss'])} | {_fmt(r['final_val_loss'])} | {_fmt(r['tokens_per_sec'], 0)} "
            f"| {_fmt(r['peak_mem_allocated_gb'], 2)} | {_fmt(r.get('final_grad_cos'), 4)} "
            f"| {r['wall_time_s'] / 60:.1f} "
            f"| {'diverged' if r['diverged'] else 'ok'} |")
    return "\n".join(rows)


def _curves(results, out_path, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    for r in results:
        h = r["history"]
        arch = r.get("arch", "unknown")
        color = COLORS.get(arch, "#8a8a85")
        linestyle = "--" if arch == "midpoint" and r.get("h", 1.0) < 0.5 else "-"
        ax.plot([t / 1e6 for t in h["val_tokens"]], h["val_loss"], marker="o", markersize=5,
                linewidth=2, linestyle=linestyle, color=color,
                label=f"{r['run_name']} (B={r['batch_size']})")

    ax.set(xlabel="Tokens seen (millions)", ylabel="Validation loss", title=title)
    ax.grid(alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#52514e", labelcolor="#52514e")
    ax.xaxis.label.set_color("#52514e")
    ax.yaxis.label.set_color("#52514e")
    ax.title.set_color("#52514e")

    if results:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _bars(results, key, label, out_path):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")

    names = [r["run_name"] for r in results]
    vals = [r.get(key) or 0 for r in results]

    # Color bars by architecture
    colors = [COLORS.get(r.get("arch", "unknown"), "#8a8a85") for r in results]
    bars = ax.barh(names, vals, color=colors, height=0.6)

    # Add value labels with dark text
    ax.bar_label(bars, fmt="%.2f" if vals and max(vals) < 100 else "%.0f",
                 padding=3, color="#1a1a19", fontsize=9)

    ax.set(xlabel=label)
    ax.invert_yaxis()
    ax.grid(alpha=0.3, linewidth=0.5, axis="x")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors="#52514e", labelcolor="#52514e")
    ax.xaxis.label.set_color("#52514e")
    ax.yaxis.label.set_color("#52514e")

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
