from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 512
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 256
    arch: str = "baseline"          # "baseline" | "euler" | "midpoint"
    h: float | None = None          # reversible step size; None -> variant default
    chunked_loss: bool = True       # compute LM head + CE in checkpointed chunks (all runs)
    loss_chunk_tokens: int = 2048

    def __post_init__(self):
        if self.h is None:
            self.h = 0.5 if self.arch == "midpoint" else 1.0


@dataclass
class TrainConfig:
    run_name: str = "baseline"
    total_tokens: int = 50_000_000
    batch_size: int = 64
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    log_every_tokens: int = 500_000
    eval_every_tokens: int = 2_500_000
    eval_tokens: int = 500_000
    seed: int = 1337
    data_dir: str = "data"
    out_dir: str = "results"
