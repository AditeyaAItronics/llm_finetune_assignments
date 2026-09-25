import math

import torch

from revllm.config import ModelConfig
from revllm.model import GPT


def test_param_count_is_about_20M():
    n = GPT(ModelConfig()).num_params()
    assert 18e6 < n < 21e6, n


def test_initial_loss_is_near_uniform(tiny_cfg):
    torch.manual_seed(0)
    m = GPT(tiny_cfg())
    x, y = torch.randint(97, (2, 16)), torch.randint(97, (2, 16))
    _, loss = m(x, y)
    assert abs(loss.item() - math.log(97)) < 0.3


def test_forward_without_targets_returns_logits(tiny_cfg):
    m = GPT(tiny_cfg())
    logits, loss = m(torch.randint(97, (2, 16)))
    assert logits.shape == (2, 16, 97) and loss is None


def test_chunked_loss_matches_full_loss_and_grads(tiny_cfg):
    torch.manual_seed(0)
    m = GPT(tiny_cfg(loss_chunk_tokens=5)).double()
    x, y = torch.randint(97, (3, 16)), torch.randint(97, (3, 16))
    _, l_chunked = m(x, y)
    l_chunked.backward()
    g_chunked = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    m.cfg.chunked_loss = False
    _, l_full = m(x, y)
    l_full.backward()
    assert torch.allclose(l_chunked, l_full, rtol=1e-10)
    for g, p in zip(g_chunked, m.parameters()):
        assert torch.allclose(g, p.grad, rtol=1e-8, atol=1e-12)
