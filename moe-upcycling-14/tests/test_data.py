import torch
from src.data import CharData


def test_vocab_and_split():
    d = CharData()
    assert d.vocab_size == 65
    total = d.train.numel() + d.val.numel()
    assert total == 1115394
    assert abs(d.val.numel() / total - 0.1) < 1e-3


def test_batch_shapes_and_shift():
    d = CharData()
    g = torch.Generator().manual_seed(0)
    x, y = d.batch("train", 8, 16, g)
    assert x.shape == y.shape == (8, 16)
    assert x.dtype == torch.long
    assert torch.equal(x[:, 1:], y[:, :-1])  # y is x shifted by one


def test_batch_is_deterministic():
    d = CharData()
    a = d.batch("val", 4, 32, torch.Generator().manual_seed(5))
    b = d.batch("val", 4, 32, torch.Generator().manual_seed(5))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_decode_roundtrip():
    d = CharData()
    assert d.decode(d.train[:14].tolist()) == "First Citizen:"
