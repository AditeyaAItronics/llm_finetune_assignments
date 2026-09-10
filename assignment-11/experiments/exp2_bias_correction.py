"""Q2: bias correction on vs off. First 20 steps plotted both ways.

THE COUNTER-INTUITIVE PART, and the real content of this question: removing bias
correction makes the early steps LARGER, not smaller. Both moments start at zero,
but v is biased toward zero far harder than m (b2=0.999 vs b1=0.9) and sqrt(v)
sits in the denominator. The net factor on the step size is

    r(t) = sqrt(1 - b2**t) / (1 - b1**t)

which is 0.316 at t=1, DIPS to ~0.152 near t=12 (uncorrected steps are then ~6.6x
too large), and only then climbs slowly back toward 1. This is precisely why
uncorrected Adam needs warmup - warmup is compensating for the inflated early
steps that bias correction would have removed for free.

Two definitions of "stops mattering", both reported, because the question has no
single answer and quoting one number without its definition is not an answer:

  t_star_stepsize : first t after which r(t) stays within 1% of 1.0 - i.e. when
                    the correction stops changing the size of the step Adam takes.
  t_star_visible  : first t after which the two weight trajectories stay within
                    1% of each other in cumulative displacement.

The second is far larger than the first, and that is not a bug. r(t) < 1 for EVERY
t, so the uncorrected run is permanently slightly ahead: the absolute displacement
gap accumulates to a constant and only washes out as total displacement grows. The
plot window is 20 steps as the brief asks, but the measurement window is not - at
step 20 the trajectories are still far apart, so reading t* off a 20-step plot
would report the edge of the window rather than a property of the optimizer.
"""
import csv
import json
import math
import pathlib

import torch

from src.adam_manual import ManualAdam
from src.plotting import finish, new_fig
from src.seeding import set_seed

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
W0, LR, B1, B2 = 0.5, 1e-2, 0.9, 0.999
N_PLOT, N_SIM, N_LONG, TOL = 20, 200_000, 6000, 0.01


def correction_factor(t: int) -> float:
    return math.sqrt(1 - B2 ** t) / (1 - B1 ** t)


def main() -> None:
    set_seed(1337)
    RESULTS.mkdir(exist_ok=True)

    w_on = torch.tensor([W0], dtype=torch.float64, requires_grad=True)
    w_off = torch.tensor([W0], dtype=torch.float64, requires_grad=True)
    opt_on = ManualAdam([w_on], lr=LR, betas=(B1, B2), bias_correction=True)
    opt_off = ManualAdam([w_off], lr=LR, betas=(B1, B2), bias_correction=False)

    # Simulate well past the 20-step plotting window. The brief asks to PLOT the
    # first twenty steps, but "the number of steps after which the difference
    # stops mattering" cannot be read off a 20-step window if the curves are
    # still apart at step 20 - that would report the window edge, not an answer.
    plot_rows, last_over, rel_at_plot_end = [], 0, 0.0
    gap_trace = {}
    g = torch.tensor([0.1], dtype=torch.float64)  # constant: isolates the correction
    for t in range(1, N_SIM + 1):
        prev_on, prev_off = w_on.item(), w_off.item()
        w_on.grad = g.clone()
        w_off.grad = g.clone()
        opt_on.step()
        opt_off.step()
        disp_on = abs(w_on.item() - W0)
        gap = abs(w_on.item() - w_off.item())
        rel = gap / disp_on if disp_on > 0 else 0.0
        if rel > TOL:
            last_over = t
        if t <= N_PLOT:
            plot_rows.append({
                "t": t,
                "w_corrected": w_on.item(),
                "w_uncorrected": w_off.item(),
                "step_corrected": w_on.item() - prev_on,
                "step_uncorrected": w_off.item() - prev_off,
                "ratio": correction_factor(t),
                "rel_disp_diff": rel,
            })
            if t == N_PLOT:
                rel_at_plot_end = rel
        if t in (100, 1000, 10_000, 50_000, N_SIM):
            gap_trace[t] = {"abs_gap": gap, "rel_gap": rel}

    t_star_visible = last_over + 1
    assert t_star_visible < N_SIM, (
        f"extend N_SIM: still >{TOL:.0%} apart at step {N_SIM}")

    ratios = [correction_factor(t) for t in range(1, N_LONG + 1)]
    over_long = [t for t, r in enumerate(ratios, start=1) if abs(r - 1.0) > TOL]
    t_star_stepsize = (max(over_long) + 1) if over_long else 1
    assert t_star_stepsize < N_LONG, "extend N_LONG: the factor has not converged yet"

    t_worst = min(range(1, N_LONG + 1), key=correction_factor)

    with (RESULTS / "bias_correction.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(plot_rows[0]))
        wr.writeheader()
        wr.writerows(plot_rows)

    (RESULTS / "bias_correction.json").write_text(json.dumps({
        "tolerance": TOL,
        "t_star_visible": t_star_visible,
        "t_star_stepsize": t_star_stepsize,
        "worst_step": t_worst,
        "worst_factor": correction_factor(t_worst),
        "uncorrected_inflation_at_worst": 1.0 / correction_factor(t_worst),
        "factor_at_t1": correction_factor(1),
        "n_steps_simulated": N_SIM,
        "rel_diff_at_20": rel_at_plot_end,
        "gap_trace": gap_trace,
        "direction": "uncorrected steps are LARGER than corrected ones early on",
    }, indent=2))

    fig, (ax1, ax2) = new_fig(1, 2, figsize=(12, 4.5))
    ts = [r["t"] for r in plot_rows]
    ax1.plot(ts, [r["w_corrected"] for r in plot_rows], "o-", label="bias correction ON")
    ax1.plot(ts, [r["w_uncorrected"] for r in plot_rows], "s--", label="bias correction OFF")
    ax1.set_xlim(1, N_PLOT)
    ax1.set_xlabel("step t")
    ax1.set_ylabel("weight")
    pct = rel_at_plot_end * 100
    ax1.set_title(f"First {N_PLOT} steps, constant gradient g = 0.1\n"
                  f"(still {pct:.0f}% apart at step {N_PLOT})")
    ax1.legend()

    ax2.semilogx(range(1, N_LONG + 1), ratios)
    ax2.axhline(1.0, color="k", ls="-", lw=0.8)
    ax2.axhline(0.99, color="r", ls=":", lw=0.8, label="1% band")
    ax2.axvline(t_worst, color="orange", ls="--",
                label=f"worst at t={t_worst} ({1/correction_factor(t_worst):.1f}x too large)")
    ax2.axvline(t_star_stepsize, color="k", ls=":",
                label=f"t* (step size) = {t_star_stepsize}")
    ax2.set_xlabel("step t (log scale)")
    ax2.set_ylabel(r"$\sqrt{1-\beta_2^t}\,/\,(1-\beta_1^t)$")
    ax2.set_title("Step-size correction factor (<1 means uncorrected is too big)")
    ax2.legend(fontsize=8)
    finish(fig, RESULTS / "bias_correction.png")

    print(json.dumps({
        "t_star_visible": t_star_visible,
        "t_star_stepsize": t_star_stepsize,
        "worst_step": t_worst,
        "uncorrected_inflation_at_worst": 1.0 / correction_factor(t_worst),
    }, indent=2))


if __name__ == "__main__":
    main()
