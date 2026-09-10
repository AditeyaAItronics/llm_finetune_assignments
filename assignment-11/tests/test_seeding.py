import torch

from src.seeding import set_seed


def test_set_seed_makes_torch_randn_reproducible():
    set_seed(1337)
    a = torch.randn(8)
    set_seed(1337)
    b = torch.randn(8)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    set_seed(1)
    a = torch.randn(8)
    set_seed(2)
    b = torch.randn(8)
    assert not torch.equal(a, b)
