"""Q5: LR sweep at widths 256/512/1024, minima marked, extrapolated to 4096.

Everything except width is held fixed. We fit log10(lr*) = a + b*log2(width)
through the three minima and extrapolate. With three points and no repeats the
fit is weakly determined - the JSON records an interval, and the README says so.
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
                   "parameterisation": "standard (std = 1/sqrt(fan_in))",
                   "confound": "n_heads fixed at 4, so head_dim scales with width"},
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
