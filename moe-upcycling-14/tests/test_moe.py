import torch
from src.moe import MoE


def test_shapes_and_fractions():
    torch.manual_seed(0)
    moe = MoE(16, 32, n_experts=4, top_k=2)
    y = moe(torch.randn(2, 5, 16))
    assert y.shape == (2, 5, 16)
    assert moe.last_frac.shape == (4,)
    assert abs(moe.last_frac.sum().item() - 1.0) < 1e-6


def test_matches_naive_reference():
    """Vectorised dispatch must equal a per-token loop."""
    torch.manual_seed(0)
    moe = MoE(8, 16, n_experts=4, top_k=2)
    x = torch.randn(1, 6, 8)
    y = moe(x)[0]
    h = x[0]
    probs = moe.router(h).softmax(-1)
    for i in range(6):
        v, idx = probs[i].topk(2)
        g = v / v.sum()
        ref = sum(g[j] * moe.experts[idx[j]](h[i]) for j in range(2))
        assert torch.allclose(y[i], ref, atol=1e-6)


def test_router_and_experts_get_gradients():
    torch.manual_seed(0)
    moe = MoE(16, 32, n_experts=4, top_k=2)
    y = moe(torch.randn(4, 8, 16))
    (y.pow(2).mean() + moe.aux_loss).backward()
    assert moe.router.weight.grad is not None and moe.router.weight.grad.abs().sum() > 0
    used = [e for e in moe.experts if e.fc1.weight.grad is not None]
    assert len(used) >= 2


def test_aux_loss_is_one_when_perfectly_balanced():
    moe = MoE(4, 8, n_experts=4, top_k=1)
    with torch.no_grad():
        moe.router.weight.copy_(torch.eye(4) * 50)  # token i -> expert i, prob ~1
    moe(torch.eye(4).view(1, 4, 4))
    assert abs(moe.aux_loss.item() - 1.0) < 1e-3
