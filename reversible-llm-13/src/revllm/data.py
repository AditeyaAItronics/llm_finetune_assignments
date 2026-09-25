import os

import numpy as np
import torch


def prepare_tinystories(data_dir: str, train_tokens: int = 55_000_000, val_tokens: int = 1_000_000) -> None:
    """Tokenize TinyStories with GPT-2 BPE into uint16 memmaps. Skips splits that already exist."""
    import tiktoken
    from datasets import load_dataset

    os.makedirs(data_dir, exist_ok=True)
    enc = tiktoken.get_encoding("gpt2")
    for split, hf_split, n_target in (("train", "train", train_tokens), ("val", "validation", val_tokens)):
        path = os.path.join(data_dir, f"{split}.bin")
        if os.path.exists(path):
            continue
        buf = np.empty(n_target, dtype=np.uint16)
        pos = 0
        for ex in load_dataset("roneneldan/TinyStories", split=hf_split, streaming=True):
            ids = enc.encode_ordinary(ex["text"]) + [enc.eot_token]
            take = min(len(ids), n_target - pos)
            buf[pos:pos + take] = ids[:take]
            pos += take
            if pos >= n_target:
                break
        tmp = path + ".tmp"
        buf[:pos].tofile(tmp)
        os.replace(tmp, path)  # atomic: an interrupted write never leaves a truncated .bin behind
        print(f"{split}: {pos:,} tokens -> {path}")


def load_split(data_dir: str, split: str) -> np.memmap:
    return np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")


def get_batch(data, batch_size: int, block_size: int, device: str, generator=None):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator).tolist()
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)
