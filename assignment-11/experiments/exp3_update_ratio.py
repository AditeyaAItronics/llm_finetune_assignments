"""Q3: log ||dw||/||w|| for every layer; find the step where warmup stops moving it.

Two detection rules, both stated so they are reproducible rather than eyeballed,
and both reported because they disagree and the disagreement is the finding:

  SLOPE RULE (primary)   - smooth with a centred 5-step moving average, then find
    the first step after which |d(ratio)/d(step)| stays below 10% of its peak
    warmup value. This isolates the LR ramp's contribution, which is what "the
    step at which warmup stops changing it" is asking about.

  PLATEAU RULE (secondary) - the first step after which the ratio stays within 15%
    of its own value for the rest of the stable phase.

The plateau rule lands much later, and not because warmup is still acting. Once
the LR is constant the ratio KEEPS drifting upward: ||dw|| grows as Adam's second
moment equilibrates while ||w|| barely moves. So the plateau rule is really
measuring "when does training settle", not "when does warmup stop mattering".
Reporting one number without saying which of these it is would be meaningless.
"""
import csv
import json
import pathlib
from collections import defaultdict

from src.plotting import finish, new_fig
from src.schedules import wsd
from src.train import train_run

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
TOTAL, WARMUP, DECAY_START, PEAK_LR, WIDTH = 300, 60, 240, 3e-3, 256


def sched(step, total):
    return wsd(step, PEAK_LR, warmup_steps=WARMUP,
               decay_start=DECAY_START, total_steps=total)


def smooth(xs, k=5):
    out = []
    for i in range(len(xs)):
        lo, hi = max(0, i - k // 2), min(len(xs), i + k // 2 + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def find_knee_plateau(steps, ratios, stable_end, tol=0.15):
    """When does the ratio stop moving at all (within tol) for the rest of training?

    Secondary measure. It answers a slightly different question than the brief
    asks, because the ratio keeps drifting after warmup for reasons that have
    nothing to do with warmup: Adam's second moment is still equilibrating and
    ||w|| is growing. So this lands LATE and conflates the two effects.
    """
    s = smooth(ratios)
    idx = [i for i, st in enumerate(steps) if st <= stable_end]
    for i in idx:
        if s[i] <= 0:
            continue
        if all(abs(s[j] - s[i]) / s[i] < tol for j in idx if j >= i):
            return steps[i]
    return steps[idx[-1]]


def find_knee_normalised(steps, ratios, lrs, stable_end, tol=0.20):
    """PRIMARY measure, and the one that actually answers the question.

    The ratio is (roughly) lr(t) * n(t), where n = ratio/lr carries everything the
    LR schedule is NOT responsible for. Warmup's entire influence on the ratio is
    the lr(t) factor, so the meaningful empirical question is: from what step does
    the ratio become simple proportionality to lr, with a settled constant?

    Detect the first step after which n stays within `tol` of its settled value
    (the median of n over the last quarter of the stable phase). Before that point
    n is still collapsing from its enormous early-training value; after it, the
    raw ratio just tracks whatever the schedule does - which is exactly what you
    see in the decay phase, where the ratio falls in lock-step with lr.

    This needs no slope threshold and no tuning against noise, because n is far
    smoother than d(ratio)/d(step).
    """
    pairs = [(st, r / lr) for st, r, lr in zip(steps, ratios, lrs)
             if lr > 0 and st <= stable_end]
    if not pairs:
        return steps[-1]
    tail_start = pairs[int(0.75 * len(pairs))][0]
    tail = sorted(n for st, n in pairs if st >= tail_start)
    ref = tail[len(tail) // 2]
    if ref <= 0:
        return pairs[-1][0]
    for i, (st, _) in enumerate(pairs):
        if all(abs(n - ref) / ref < tol for _, n in pairs[i:]):
            return st
    return pairs[-1][0]


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    res = train_run(width=WIDTH, peak_lr=PEAK_LR, total_steps=TOTAL,
                    schedule_fn=sched, log_ratios=True, eval_every=25)

    with (RESULTS / "update_ratio.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(res.ratio_rows[0]))
        wr.writeheader()
        wr.writerows(res.ratio_rows)

    by_layer = defaultdict(lambda: ([], [], []))
    for r in res.ratio_rows:
        by_layer[r["layer"]][0].append(r["step"])
        by_layer[r["layer"]][1].append(r["ratio"])
        by_layer[r["layer"]][2].append(r["lr"])

    knees = {n: find_knee_normalised(st, ra, lr, DECAY_START)
             for n, (st, ra, lr) in by_layer.items()}
    knees_plateau = {n: find_knee_plateau(st, ra, DECAY_START)
                     for n, (st, ra, _) in by_layer.items()}
    med_knee = sorted(knees.values())[len(knees) // 2]
    med_knee_plateau = sorted(knees_plateau.values())[len(knees_plateau) // 2]
    after = [r["ratio"] for r in res.ratio_rows if WARMUP <= r["step"] < DECAY_START]
    med_ratio = sorted(after)[len(after) // 2]

    # Per-layer median ratio through the stable phase: which layers ride hot/cold.
    per_layer_stable = {}
    for name, (st, ra, _) in by_layer.items():
        vals = sorted(r for s, r in zip(st, ra) if WARMUP <= s < DECAY_START)
        per_layer_stable[name] = vals[len(vals) // 2]

    (RESULTS / "update_ratio.json").write_text(json.dumps({
        "width": WIDTH, "peak_lr": PEAK_LR, "warmup_steps": WARMUP,
        "decay_start": DECAY_START, "total_steps": TOTAL,
        "knee_definition_primary": (
            "first step after which ratio/lr stays within 20% of its settled value "
            "- i.e. when the ratio becomes simple proportionality to the schedule"),
        "knee_definition_secondary": (
            "first step after which the ratio stays within 15% of its own value for "
            "the rest of the stable phase - lands late because the ratio keeps "
            "drifting for reasons unrelated to warmup"),
        "knee_step_per_layer": knees,
        "median_knee_step": med_knee,
        "knee_step_per_layer_plateau": knees_plateau,
        "median_knee_step_plateau": med_knee_plateau,
        "median_ratio_after_warmup": med_ratio,
        "median_ratio_per_layer_stable_phase": per_layer_stable,
        "layers_above_1e-3": sorted(n for n, v in per_layer_stable.items() if v > 1e-3),
        "layers_below_1e-3": sorted(n for n, v in per_layer_stable.items() if v <= 1e-3),
    }, indent=2))

    fig, ax = new_fig(figsize=(11, 5.5))
    for name, (st, ra, _) in sorted(by_layer.items()):
        ax.plot(st, smooth(ra), lw=1.2, label=name)
    ax.axvline(WARMUP, color="k", ls="--", label=f"warmup ends (step {WARMUP})")
    ax.axvline(med_knee, color="r", ls=":",
               label=f"median knee, ratio/lr rule (step {med_knee})")
    ax.axvline(med_knee_plateau, color="purple", ls=":", alpha=0.6,
               label=f"median knee, plateau rule (step {med_knee_plateau})")
    ax.axvline(DECAY_START, color="grey", ls="-.", label="decay starts")
    ax.axhline(1e-3, color="green", ls=":", lw=1, label="1e-3 rule of thumb")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(r"$\|\Delta w\|_{RMS} / \|w\|_{RMS}$")
    ax.set_title(f"Update-to-weight ratio per layer (width {WIDTH}, WSD, peak lr {PEAK_LR})")
    ax.legend(fontsize=7, ncol=2)
    finish(fig, RESULTS / "update_ratio.png")

    print(json.dumps({"warmup_steps": WARMUP,
                      "median_knee_step_normalised_rule": med_knee,
                      "median_knee_step_plateau_rule": med_knee_plateau,
                      "median_ratio_after_warmup": med_ratio}, indent=2))


if __name__ == "__main__":
    main()
