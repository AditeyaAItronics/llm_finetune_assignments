"""Q4: same model, 300 steps, cosine vs WSD, both stopped at step 200.

The brief's warning applies here: tune BOTH schedules before comparing. We sweep
the peak LR for each schedule over the same grid, pick each schedule's own best,
and only then compare. A third run answers what the assignment is really poking
at: WSD lets you decide at step 200 to spend a short decay; cosine cannot.
"""
import json
import pathlib

from src.plotting import finish, new_fig
from src.schedules import cosine_with_warmup, wsd
from src.train import train_run

RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
WIDTH, TOTAL, WARMUP, STOP = 256, 300, 30, 200
LR_GRID = [3e-4, 1e-3, 3e-3, 6e-3, 1e-2]
SEEDS = [1337, 2024, 7]  # within-schedule spread, to size the between-schedule gap


def cosine_sched(peak):
    return lambda step, total: cosine_with_warmup(
        step, peak, warmup_steps=WARMUP, total_steps=total, min_lr_frac=0.1)


def wsd_sched(peak, decay_start=240, total_override=TOTAL):
    return lambda step, total: wsd(
        step, peak, warmup_steps=WARMUP, decay_start=decay_start,
        total_steps=total_override, min_lr_frac=0.0)


def tune(make_sched):
    """Sweep peak LR over the full 300-step horizon; return {lr: final_loss}."""
    out = {}
    for lr in LR_GRID:
        r = train_run(width=WIDTH, peak_lr=lr, total_steps=TOTAL,
                      schedule_fn=make_sched(lr), eval_every=25)
        out[lr] = r.final_loss
        print(f"  peak_lr={lr:g}  final_loss={r.final_loss:.4f}")
    return out


def main() -> None:
    RESULTS.mkdir(exist_ok=True)

    print("tuning cosine:")
    cos_tune = tune(cosine_sched)
    print("tuning wsd:")
    wsd_tune = tune(lambda lr: wsd_sched(lr))

    best_cos = min(cos_tune, key=cos_tune.get)
    best_wsd = min(wsd_tune, key=wsd_tune.get)
    print(f"best cosine peak_lr={best_cos:g}   best wsd peak_lr={best_wsd:g}")

    cos = train_run(width=WIDTH, peak_lr=best_cos, total_steps=TOTAL,
                    schedule_fn=cosine_sched(best_cos), eval_every=10)
    wsd_run = train_run(width=WIDTH, peak_lr=best_wsd, total_steps=TOTAL,
                        schedule_fn=wsd_sched(best_wsd), eval_every=10)
    # The honest comparison: WSD that decides at 200 to decay over 200 -> 240.
    wsd_early = train_run(width=WIDTH, peak_lr=best_wsd, total_steps=240,
                          schedule_fn=wsd_sched(best_wsd, decay_start=200,
                                                total_override=240),
                          eval_every=10)

    # Seed spread at each schedule's own best LR. Without this the step-200
    # comparison is unfalsifiable: a gap smaller than the within-schedule spread
    # is not a result, and saying so is the correct answer to this question.
    print("measuring seed spread:")
    spread = {}
    for name, make, best in (("cosine", cosine_sched, best_cos),
                             ("wsd", wsd_sched, best_wsd)):
        vals = [train_run(width=WIDTH, peak_lr=best, total_steps=TOTAL,
                          schedule_fn=make(best), seed=s,
                          eval_every=10).loss_at(STOP) for s in SEEDS]
        spread[name] = {"seeds": SEEDS, "losses": vals,
                        "range": max(vals) - min(vals),
                        "mean": sum(vals) / len(vals)}
        print(f"  {name}: {['%.4f' % v for v in vals]} range={spread[name]['range']:.4f}")

    gap = abs(cos.loss_at(STOP) - wsd_run.loss_at(STOP))
    worst_spread = max(spread[k]["range"] for k in spread)

    payload = {
        "config": {"width": WIDTH, "total_steps": TOTAL, "warmup": WARMUP,
                   "report_at_step": STOP, "lr_grid": LR_GRID, "seeds": SEEDS},
        "tuning": {"cosine": cos_tune, "wsd": wsd_tune},
        "best_peak_lr": {"cosine": best_cos, "wsd": best_wsd},
        "loss_at_200": {"cosine": cos.loss_at(STOP), "wsd": wsd_run.loss_at(STOP)},
        "loss_at_300": {"cosine": cos.final_loss, "wsd": wsd_run.final_loss},
        "lr_at_200": {"cosine": cos.lrs[STOP - 1], "wsd": wsd_run.lrs[STOP - 1]},
        "lr_at_200_as_frac_of_peak": {
            "cosine": cos.lrs[STOP - 1] / best_cos,
            "wsd": wsd_run.lrs[STOP - 1] / best_wsd,
        },
        "wsd_decay_from_200": {"final_step": 240, "loss": wsd_early.final_loss},
        "seed_spread": spread,
        "gap_at_200": gap,
        "gap_exceeds_seed_noise": gap > worst_spread,
    }
    (RESULTS / "schedules.json").write_text(json.dumps(payload, indent=2))

    fig, (ax1, ax2) = new_fig(1, 2, figsize=(12, 4.5))
    ax1.plot(range(1, len(cos.lrs) + 1), cos.lrs, label=f"cosine (peak {best_cos:g})")
    ax1.plot(range(1, len(wsd_run.lrs) + 1), wsd_run.lrs, label=f"WSD (peak {best_wsd:g})")
    ax1.plot(range(1, len(wsd_early.lrs) + 1), wsd_early.lrs, ls="--",
             label="WSD, decay from 200")
    ax1.axvline(STOP, color="k", ls=":", label="stop at 200")
    ax1.set_xlabel("step")
    ax1.set_ylabel("learning rate")
    ax1.set_title("Schedules")
    ax1.legend(fontsize=8)

    ax2.plot(cos.steps, cos.losses, label="cosine")
    ax2.plot(wsd_run.steps, wsd_run.losses, label="WSD")
    ax2.plot(wsd_early.steps, wsd_early.losses, ls="--", label="WSD, decay from 200")
    ax2.axvline(STOP, color="k", ls=":", label="stop at 200")
    ax2.set_xlabel("step")
    ax2.set_ylabel("eval loss")
    ax2.set_title("Loss, best peak LR for each schedule")
    ax2.legend(fontsize=8)
    finish(fig, RESULTS / "schedules.png")

    print(json.dumps({"loss_at_200": payload["loss_at_200"],
                      "gap_at_200": gap,
                      "worst_seed_range": worst_spread,
                      "gap_exceeds_seed_noise": payload["gap_exceeds_seed_noise"]},
                     indent=2))


if __name__ == "__main__":
    main()
