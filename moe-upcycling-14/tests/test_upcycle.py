import torch
from src.model import GPT, GPTConfig, MLP
from src.moe import MoE, upcycle

TINY = GPTConfig(vocab_size=65, seq_len=16, d_model=32, n_layers=2, n_heads=2, d_ff=64)


def _trained_like_dense():
    torch.manual_seed(0)
    m = GPT(TINY)
    with torch.no_grad():  # make FFN weights non-trivial, like a trained model
        for p in m.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    return m.eval()


def test_upcycle_is_function_preserving():
    dense = _trained_like_dense()
    moe = upcycle(dense).eval()
    x = torch.randint(0, 65, (4, 16))
    ld, _ = dense(x)
    lm, _ = moe(x)
    assert torch.allclose(ld, lm, atol=1e-5)


def test_structure_and_param_count():
    dense = _trained_like_dense()
    moe = upcycle(dense, n_experts=4)
    assert all(isinstance(b.ffn, MoE) for b in moe.blocks)
    assert all(isinstance(b.ffn, MLP) for b in dense.blocks)  # original untouched
    ffn = sum(p.numel() for p in dense.blocks[0].ffn.parameters())
    router = TINY.d_model * 4
    expected = dense.num_params() + TINY.n_layers * (3 * ffn + router)
    assert moe.num_params() == expected


def test_experts_are_independent_copies():
    dense = _trained_like_dense()
    moe = upcycle(dense)
    e0, e1 = moe.blocks[0].ffn.experts[0], moe.blocks[0].ffn.experts[1]
    assert torch.equal(e0.fc1.weight, e1.fc1.weight)
    with torch.no_grad():
        e0.fc1.weight.add_(1.0)
    assert not torch.equal(e0.fc1.weight, e1.fc1.weight)
    assert not torch.equal(e0.fc1.weight, dense.blocks[0].ffn.fc1.weight)


def test_router_spreads_tokens_at_init():
    dense = _trained_like_dense()
    moe = upcycle(dense)
    moe(torch.randint(0, 65, (8, 16)))
    frac = moe.blocks[0].ffn.last_frac
    assert (frac > 0).sum() >= 3  # random router must not send everything to one expert
