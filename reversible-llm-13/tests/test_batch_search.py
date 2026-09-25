import pytest

from revllm import batch_search
from revllm.config import ModelConfig


def test_finds_exact_boundary_and_safe_recommendation(monkeypatch):
    monkeypatch.setattr(batch_search, "fits", lambda mc, b, steps=2, **kw: b <= 100)
    r = batch_search.find_max_batch(ModelConfig())
    assert r["max_batch"] == 100 and r["first_oom"] == 101
    assert r["recommended"] == 88  # 90% of 100, rounded down to a multiple of 8
    assert r["use_rev_fn"] is True


def test_respects_limit(monkeypatch):
    monkeypatch.setattr(batch_search, "fits", lambda mc, b, steps=2, **kw: True)
    assert batch_search.find_max_batch(ModelConfig(), limit=64)["max_batch"] == 64


def test_raises_when_nothing_fits(monkeypatch):
    monkeypatch.setattr(batch_search, "fits", lambda mc, b, steps=2, **kw: False)
    with pytest.raises(RuntimeError):
        batch_search.find_max_batch(ModelConfig())


def test_find_max_batch_passes_use_rev_fn(monkeypatch):
    seen = []
    monkeypatch.setattr(batch_search, "fits", lambda mc, b, steps=2, **kw: seen.append(kw) or b <= 10)
    r = batch_search.find_max_batch(ModelConfig(), start=4, use_rev_fn=False)
    assert r["use_rev_fn"] is False
    assert all(kw.get("use_rev_fn") is False for kw in seen)


def test_is_oom_detects_cuda_oom_and_alloc_failed_messages():
    assert batch_search._is_oom(RuntimeError("CUDA out of memory. Tried to allocate ..."))
    assert batch_search._is_oom(RuntimeError("cuDNN error: ALLOC_FAILED"))
    assert batch_search._is_oom(RuntimeError("Out Of Memory"))  # case-insensitive
    assert not batch_search._is_oom(RuntimeError("some other runtime error"))
    assert not batch_search._is_oom(ValueError("not even a RuntimeError"))


def test_fits_reraises_non_oom_runtime_error(monkeypatch):
    class FakeModel:
        def cuda(self):
            return self

        def parameters(self):
            return iter([])

    def boom(*a, **kw):
        raise RuntimeError("some unrelated failure")

    monkeypatch.setattr(batch_search, "GPT", lambda mc: FakeModel())
    monkeypatch.setattr(batch_search.torch.optim, "AdamW", lambda *a, **kw: boom())
    with pytest.raises(RuntimeError, match="unrelated failure"):
        batch_search.fits(ModelConfig(), 8)
