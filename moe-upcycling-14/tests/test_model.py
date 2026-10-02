import math

import torch
from src.model import GPT, GPTConfig

TINY = GPTConfig(vocab_size=65, seq_len=16, d_model=32, n_layers=2, n_heads=2, d_ff=64)


def test_forward_shapes_and_initial_loss():
    torch.manual_seed(0)
    m = GPT(TINY)
    x = torch.randint(0, 65, (3, 16))
    logits, loss = m(x, x)
    assert logits.shape == (3, 16, 65)
    # freshly initialised model should be near uniform: ln(65) ~= 4.17
    assert abs(loss.item() - math.log(65)) < 0.5


def test_causality():
    torch.manual_seed(0)
    m = GPT(TINY).eval()
    x = torch.randint(0, 65, (1, 16))
    x2 = x.clone()
    x2[0, 10] = (x2[0, 10] + 1) % 65
    l1, _ = m(x)
    l2, _ = m(x2)
    assert torch.allclose(l1[0, :10], l2[0, :10], atol=1e-6)  # future token must not leak back


def test_dense_aux_loss_is_zero():
    m = GPT(TINY)
    assert m.aux_loss().item() == 0.0


def test_default_param_count():
    n = GPT(GPTConfig()).num_params()
    assert 3_000_000 < n < 3_500_000
