import math

import torch

from src.adam_manual import ManualAdam, adam_step_scalar


def test_scalar_trace_matches_closed_form_first_step():
    out = adam_step_scalar(
        w=0.5, g=0.1, m=0.0, v=0.0, t=1,
        lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, bias_correction=True,
    )
    # rel_tol is 1e-12, not 1e-15: (1.0 - 0.999) is 0.0010000000000000009 in
    # binary floating point, so a tighter tolerance tests the representation of
    # the constants rather than the arithmetic.
    assert math.isclose(out["m"], 0.1 * 0.1, rel_tol=1e-12)
    assert math.isclose(out["v"], 0.001 * 0.01, rel_tol=1e-12)
    assert math.isclose(out["m_hat"], out["m"] / (1 - 0.9 ** 1), rel_tol=1e-12)
    assert math.isclose(out["v_hat"], out["v"] / (1 - 0.999 ** 1), rel_tol=1e-12)
    expected_update = -1e-3 * out["m_hat"] / (math.sqrt(out["v_hat"]) + 1e-8)
    assert math.isclose(out["update"], expected_update, rel_tol=1e-12)


def test_first_adam_step_is_minus_lr_times_sign_of_gradient():
    """The headline intuition: step 1 ignores gradient magnitude entirely."""
    for g in (0.1, 5.0, 1e4):
        out = adam_step_scalar(0.0, g, 0.0, 0.0, 1, 1e-3, 0.9, 0.999, 1e-8, True)
        assert math.isclose(out["update"], -1e-3, rel_tol=1e-6)


def test_manual_adam_matches_torch_adam_over_five_steps_float64():
    grads = [0.1, -0.3, 0.05, 0.2, -0.15]

    mine = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    theirs = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)

    opt_mine = ManualAdam([mine], lr=1e-3)
    opt_theirs = torch.optim.Adam([theirs], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    for g in grads:
        mine.grad = torch.tensor([g], dtype=torch.float64)
        theirs.grad = torch.tensor([g], dtype=torch.float64)
        opt_mine.step()
        opt_theirs.step()
        assert abs(mine.item() - theirs.item()) < 1e-12


def test_manual_adam_moments_match_torch_state():
    """The brief says check EACH quantity, not just the resulting weight."""
    grads = [0.1, -0.3, 0.05]

    mine = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    theirs = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    opt_mine = ManualAdam([mine], lr=1e-3)
    opt_theirs = torch.optim.Adam([theirs], lr=1e-3, betas=(0.9, 0.999), eps=1e-8)

    for g in grads:
        mine.grad = torch.tensor([g], dtype=torch.float64)
        theirs.grad = torch.tensor([g], dtype=torch.float64)
        opt_mine.step()
        opt_theirs.step()
        st = opt_theirs.state[theirs]
        assert abs(opt_mine.state[mine]["m"].item() - st["exp_avg"].item()) < 1e-15
        assert abs(opt_mine.state[mine]["v"].item() - st["exp_avg_sq"].item()) < 1e-15


def test_bias_correction_off_takes_a_LARGER_first_step():
    """Counter-intuitive but correct, and the heart of Q2.

    Both moments start at zero, but v is biased toward zero far harder than m
    (b2=0.999 vs b1=0.9) and sqrt(v) is in the DENOMINATOR. Net effect on the
    step size is sqrt(1-b2**t)/(1-b1**t) = 0.316 at t=1, so removing bias
    correction INFLATES the early steps rather than shrinking them.
    """
    w_on = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    w_off = torch.tensor([0.5], dtype=torch.float64, requires_grad=True)
    on = ManualAdam([w_on], lr=1e-3, bias_correction=True)
    off = ManualAdam([w_off], lr=1e-3, bias_correction=False)
    for w, opt in ((w_on, on), (w_off, off)):
        w.grad = torch.tensor([0.1], dtype=torch.float64)
        opt.step()
    step_on = abs(w_on.item() - 0.5)
    step_off = abs(w_off.item() - 0.5)
    assert step_off > step_on
    assert math.isclose(step_on / step_off, math.sqrt(1 - 0.999) / (1 - 0.9),
                        rel_tol=1e-4)


def test_eps_placement_matters():
    """Guard against sqrt(v_hat + eps): it is wrong, and this pins the difference."""
    correct = adam_step_scalar(0.5, 1e-6, 0.0, 0.0, 1, 1e-3, 0.9, 0.999, 1e-8, True)
    wrong_denom = math.sqrt(correct["v_hat"] + 1e-8)
    wrong_update = -1e-3 * correct["m_hat"] / wrong_denom
    assert abs(correct["update"] - wrong_update) > 1e-6
