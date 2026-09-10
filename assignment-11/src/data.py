"""A deterministic, dependency-free token stream.

We do NOT download a corpus: graders must be able to run this offline and get
identical numbers. The stream is a synthetic mixture of repeated motifs plus
noise, which gives a loss curve with real structure (learnable, not trivial).
"""
import torch

VOCAB_SIZE = 256


def make_dataset(seq_len: int = 128, n_tokens: int = 200_000, seed: int = 1337):
    g = torch.Generator().manual_seed(seed)
    motifs = [torch.randint(0, VOCAB_SIZE, (k,), generator=g) for k in (3, 5, 8, 13)]
    out, total = [], 0
    while total < n_tokens:
        if torch.rand(1, generator=g).item() < 0.7:
            m = motifs[torch.randint(0, len(motifs), (1,), generator=g).item()]
            out.append(m)
            total += m.numel()
        else:
            out.append(torch.randint(0, VOCAB_SIZE, (4,), generator=g))
            total += 4
    tokens = torch.cat(out)[:n_tokens].long()
    assert tokens.numel() > seq_len + 1
    return tokens, VOCAB_SIZE


def get_batch(tokens: torch.Tensor, batch_size: int, seq_len: int,
              generator: torch.Generator, device: str = "cpu"):
    hi = tokens.numel() - seq_len - 1
    idx = torch.randint(0, hi, (batch_size,), generator=generator)
    x = torch.stack([tokens[i:i + seq_len] for i in idx])
    y = torch.stack([tokens[i + 1:i + seq_len + 1] for i in idx])
    return x.to(device), y.to(device)
