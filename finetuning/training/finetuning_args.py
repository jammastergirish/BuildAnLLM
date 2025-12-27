"""Training arguments for fine-tuning."""

from dataclasses import dataclass


@dataclass
class FinetuningArgs:
    """Arguments for supervised fine-tuning."""
    batch_size: int = 4
    epochs: int = 3
    max_steps_per_epoch: int = 1000
    lr: float = 1e-5  # Lower than pre-training
    weight_decay: float = 0.01
    save_dir: str = None
    save_interval: int = 500
    eval_iters: int = 50
    warmup_steps: int = 100  # Optional: learning rate warmup
    # LoRA parameters
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 8.0
    lora_dropout: float = 0.0
    lora_target_modules: str = "all"  # "all", "attention", "mlp", or list

