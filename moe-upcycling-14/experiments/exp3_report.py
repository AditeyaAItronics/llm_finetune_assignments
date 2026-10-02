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
