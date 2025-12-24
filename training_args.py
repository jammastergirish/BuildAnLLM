from dataclasses import dataclass


@dataclass
class TransformerTrainingArgs:
    batch_size: int = 32
    epochs: int = 10
    max_steps_per_epoch: int = 500
    lr: float = 1e-3
    weight_decay: float = 1e-2
    save_dir: str = "checkpoints"
    save_interval: int = 1000  # Save checkpoint every N iterations
    eval_iters: int = 200  # Number of iterations for evaluation
