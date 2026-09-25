import pytest
import torch

from revllm.config import ModelConfig
from revllm.model import GPT
from revllm.reversible import grad_agreement, make_rev_blocks, reconstruction_error

VARIANTS = ["euler", "midpoint"]


@pytest.mark.parametrize("arch", VARIANTS)
def test_block_inverse_reconstructs_inputs(tiny_cfg, arch):
    torch.manual_seed(0)
    blk = make_rev_blocks(tiny_cfg(arch))[0].double()
    x1 = torch.randn(2, 16, 32, dtype=torch.float64)
    x2 = torch.randn(2, 16, 32, dtype=torch.float64)
    r1, r2 = blk.inverse(*blk(x1, x2))
    assert torch.allclose(r1, x1, atol=1e-10) and torch.allclose(r2, x2, atol=1e-10)


@pytest.mark.parametrize("arch", VARIANTS)
def test_memory_saving_backward_matches_plain_autograd(tiny_cfg, arch):
    torch.manual_seed(0)
    m = GPT(tiny_cfg(arch)).double()
    x, y = torch.randint(97, (2, 16)), torch.randint(97, (2, 16))
    _, l_ref = m(x, y, use_rev_fn=False)
    l_ref.backward()
    ref = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    _, l_rev = m(x, y, use_rev_fn=True)
    l_rev.backward()
    assert torch.allclose(l_ref, l_rev, rtol=1e-12)
    for (name, p), g in zip(m.named_parameters(), ref):
        assert torch.allclose(p.grad, g, rtol=1e-6, atol=1e-10), name


@pytest.mark.parametrize("arch", VARIANTS)
def test_reconstruction_error_is_tiny_in_fp64(tiny_cfg, arch):
    torch.manual_seed(0)
    m = GPT(tiny_cfg(arch)).double()
    assert reconstruction_error(m.blocks, torch.randn(2, 16, 32, dtype=torch.float64)) < 1e-9


@pytest.mark.parametrize("arch", VARIANTS)
def test_grad_agreement_matches_in_fp64(tiny_cfg, arch):
    torch.manual_seed(0)
    m = GPT(tiny_cfg(arch)).double()
    x, y = torch.randint(97, (2, 16)), torch.randint(97, (2, 16))
    out = grad_agreement(m, x, y)
    assert out["grad_cos"] > 0.999999
    assert out["grad_rel_err"] < 1e-6
    assert all(p.grad is None for p in m.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("arch", VARIANTS)
def test_reversible_uses_less_activation_memory(arch):
    cfg = ModelConfig(arch=arch, vocab_size=512)  # small vocab so block activations dominate

    def activation_peak(use_rev):
        torch.manual_seed(0)
        m = GPT(cfg).cuda()
        x = torch.randint(cfg.vocab_size, (8, cfg.block_size), device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        _, loss = m(x, x, use_rev_fn=use_rev)
        loss.backward()
        return torch.cuda.max_memory_allocated() - base

    assert activation_peak(True) < 0.5 * activation_peak(False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("arch", VARIANTS)
def test_reversible_grads_survive_fp16_autocast(arch):
    cfg = ModelConfig(arch=arch, vocab_size=512)
    torch.manual_seed(0)
    m = GPT(cfg).cuda()
    x = torch.randint(cfg.vocab_size, (4, cfg.block_size), device="cuda")
    y = torch.randint(cfg.vocab_size, (4, cfg.block_size), device="cuda")
    block_params = [p for p in m.blocks.parameters()]

    def grads(use_rev_fn):
        with torch.autocast("cuda", dtype=torch.float16):
            _, loss = m(x, y, use_rev_fn=use_rev_fn)
        return torch.autograd.grad(loss, block_params, allow_unused=True)

    g_plain = grads(False)
    g_rev = grads(True)
    for g in g_rev:
        assert g is not None and not torch.all(g == 0)

    flat_plain = torch.cat([g.reshape(-1).float() for g, p in zip(g_plain, block_params)
                             if g is not None] or [torch.zeros(0)])
    flat_rev = torch.cat([g.reshape(-1).float() for g in g_rev])
    rel_err = (flat_rev - flat_plain).norm().item() / flat_plain.norm().item()
    assert rel_err < 0.05
