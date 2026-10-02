"""Stage 1: train the dense ("Linear") GPT from scratch."""
import platform
import sys
from dataclasses import asdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data import CharData  # noqa: E402
from src.model import GPT, GPTConfig  # noqa: E402
from src.seeding import set_seed  # noqa: E402
from src.train import TrainConfig, train  # noqa: E402

DENSE_STEPS = 1500


def main():
    torch.set_num_threads(16)
    set_seed(1337)
    data = CharData()
    cfg = GPTConfig(vocab_size=data.vocab_size)
    model = GPT(cfg)
    tcfg = TrainConfig(run_name="dense_pretrain", steps=DENSE_STEPS, lr=1e-3, min_lr=1e-4, warmup=100)
    train(model, data, tcfg, ROOT / "logs")
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "steps": DENSE_STEPS},
               ROOT / "checkpoints" / "dense.pt")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "environment.txt").write_text(
        f"python {platform.python_version()}\ntorch {torch.__version__}\n"
        f"device cpu, threads {torch.get_num_threads()}\nplatform {platform.platform()}\n")


if __name__ == "__main__":
    main()
