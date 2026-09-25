import numpy as np
import torch

from revllm.config import ModelConfig
from revllm.data import get_batch, load_split


def test_get_batch_targets_are_inputs_shifted_by_one(tmp_path):
    np.arange(1000, dtype=np.uint16).tofile(tmp_path / "train.bin")
    data = load_split(str(tmp_path), "train")
    x, y = get_batch(data, 4, 8, "cpu", torch.Generator().manual_seed(0))
    assert x.shape == (4, 8) and x.dtype == torch.int64
    assert torch.equal(y, x + 1)


def test_step_size_defaults_depend_on_variant():
    assert ModelConfig(arch="euler").h == 1.0
    assert ModelConfig(arch="midpoint").h == 0.5
    assert ModelConfig(arch="midpoint", h=0.25).h == 0.25
