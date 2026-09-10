import math

from src.schedules import cosine_with_warmup, wsd


def test_cosine_warmup_is_linear_and_hits_peak():
    assert cosine_with_warmup(0, 1e-3, 100, 1000) == 0.0
    assert math.isclose(cosine_with_warmup(50, 1e-3, 100, 1000), 5e-4)
    assert math.isclose(cosine_with_warmup(100, 1e-3, 100, 1000), 1e-3)


def test_cosine_decays_to_min_lr_at_horizon():
    lr = cosine_with_warmup(1000, 1e-3, 100, 1000, min_lr_frac=0.1)
    assert math.isclose(lr, 1e-4, rel_tol=1e-9)


def test_cosine_truncated_at_200_of_300_is_still_well_above_min():
    lr = cosine_with_warmup(200, 1e-3, 30, 300, min_lr_frac=0.1)
    assert 2e-4 < lr < 6e-4


def test_wsd_is_flat_through_the_stable_phase():
    a = wsd(120, 1e-3, warmup_steps=30, decay_start=240, total_steps=300)
    b = wsd(200, 1e-3, warmup_steps=30, decay_start=240, total_steps=300)
    assert math.isclose(a, 1e-3) and math.isclose(b, 1e-3)


def test_wsd_decays_to_zero_at_horizon():
    assert math.isclose(wsd(300, 1e-3, 30, 240, 300, min_lr_frac=0.0), 0.0, abs_tol=1e-12)


def test_schedules_never_exceed_peak_or_go_negative():
    for step in range(0, 400):
        c = cosine_with_warmup(step, 1e-3, 30, 300, min_lr_frac=0.1)
        w = wsd(step, 1e-3, 30, 240, 300)
        assert 0.0 <= c <= 1e-3 + 1e-15
        assert 0.0 <= w <= 1e-3 + 1e-15
