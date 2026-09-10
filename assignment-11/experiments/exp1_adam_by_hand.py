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
    assert max_abs_diff < 1e-12, f"disagreement {max_abs_diff:.3e} - check eps placement"

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
