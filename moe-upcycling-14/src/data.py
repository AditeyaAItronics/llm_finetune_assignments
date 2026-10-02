"""Character-level Tiny Shakespeare. Committed to the repo so runs are offline and exact."""
from pathlib import Path

import torch

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"


class CharData:
    def __init__(self, path: Path = DATA_PATH, val_frac: float = 0.1):
        text = Path(path).read_text(encoding="utf-8")
        self.chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(ids) * (1 - val_frac))
        self.train, self.val = ids[:n], ids[n:]
        self.vocab_size = len(self.chars)

    def batch(self, split: str, batch_size: int, seq_len: int, generator: torch.Generator):
        src = self.train if split == "train" else self.val
        ix = torch.randint(0, src.numel() - seq_len - 1, (batch_size,), generator=generator)
        x = torch.stack([src[i:i + seq_len] for i in ix])
        y = torch.stack([src[i + 1:i + 1 + seq_len] for i in ix])
        return x, y

    def decode(self, ids: list[int]) -> str:
        return "".join(self.chars[i] for i in ids)
