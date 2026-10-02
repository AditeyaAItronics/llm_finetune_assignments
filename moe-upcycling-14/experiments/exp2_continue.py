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
        (ROOT / "logs").mkdir(exist_ok=True)
        (ROOT / "logs" / "conversion_check.json").write_text(json.dumps(check, indent=2))
        print("conversion check:", check)
        model = moe
    train(model, data, tcfg, ROOT / "logs")
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "arch": args.arch},
               ROOT / "checkpoints" / f"{args.arch}_continue.pt")


if __name__ == "__main__":
    main()
