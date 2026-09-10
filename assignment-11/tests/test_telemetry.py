import torch

from src.telemetry import UpdateRatioLogger


def test_ratio_is_zero_when_nothing_moves():
    model = torch.nn.Linear(4, 4)
    log = UpdateRatioLogger(model)
    log.snapshot()
    log.record(step=1, lr=1e-3)
    assert all(r["ratio"] == 0.0 for r in log.rows)


def test_ratio_matches_hand_computation():
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)          # w_rms = 1.0
    log = UpdateRatioLogger(model)
    log.snapshot()
    with torch.no_grad():
        model.weight.add_(0.1)           # d_rms = 0.1
    log.record(step=1, lr=1e-3)
    row = [r for r in log.rows if r["layer"] == "weight"][0]
    assert abs(row["ratio"] - 0.1) < 1e-6


def test_one_row_per_trainable_tensor_per_step():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 3))
    log = UpdateRatioLogger(model)
    n_params = sum(1 for _ in model.parameters())
    for step in range(1, 4):
        log.snapshot()
        log.record(step=step, lr=1e-3)
    assert len(log.rows) == 3 * n_params


def test_snapshot_is_a_copy_not_a_reference():
    """If snapshot aliased the live tensor every ratio would come out zero."""
    model = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.fill_(1.0)
    log = UpdateRatioLogger(model)
    log.snapshot()
    with torch.no_grad():
        model.weight.mul_(2.0)
    log.record(step=1, lr=1e-3)
    assert log.rows[0]["ratio"] > 0.5
