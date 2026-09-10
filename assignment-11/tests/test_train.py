from src.schedules import cosine_with_warmup
from src.train import train_run


def _sched(step, total):
    return cosine_with_warmup(step, 1e-3, warmup_steps=5, total_steps=total)


def test_short_run_reduces_loss():
    r = train_run(width=64, peak_lr=1e-3, total_steps=40,
                  schedule_fn=_sched, batch_size=8, seq_len=32)
    assert r.losses[0] > r.losses[-1]


def test_run_is_deterministic():
    kw = dict(width=64, peak_lr=1e-3, total_steps=20,
              schedule_fn=_sched, batch_size=8, seq_len=32)
    a = train_run(**kw)
    b = train_run(**kw)
    assert a.final_loss == b.final_loss


def test_different_seeds_give_different_results():
    kw = dict(width=64, peak_lr=1e-3, total_steps=20,
              schedule_fn=_sched, batch_size=8, seq_len=32)
    a = train_run(seed=1337, **kw)
    b = train_run(seed=2024, **kw)
    assert a.final_loss != b.final_loss


def test_ratio_rows_populated_when_requested():
    r = train_run(width=64, peak_lr=1e-3, total_steps=10, schedule_fn=_sched,
                  batch_size=8, seq_len=32, log_ratios=True)
    assert len(r.ratio_rows) > 0
    assert {"step", "layer", "ratio", "lr"} <= set(r.ratio_rows[0])


def test_loss_at_returns_the_exact_logged_step():
    r = train_run(width=64, peak_lr=1e-3, total_steps=20, schedule_fn=_sched,
                  batch_size=8, seq_len=32, eval_every=10)
    assert 10 in r.steps
    assert r.loss_at(10) == r.losses[r.steps.index(10)]
