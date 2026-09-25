import pytest

from revllm.config import ModelConfig


@pytest.fixture
def tiny_cfg():
    def make(arch="baseline", **kw):
        return ModelConfig(vocab_size=97, block_size=16, n_layer=3, n_head=2, d_model=32, arch=arch, **kw)
    return make
