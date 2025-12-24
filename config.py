from dataclasses import dataclass


@dataclass
class GPTConfig:
    d_model: int = 768
    debug: bool = True
    layer_norm_eps: float = 1e-5
    d_vocab: int = 50257
    init_range: float = 0.02
    n_ctx: int = 1024
    d_head: int = 64
    d_mlp: int = 3072
    n_heads: int = 12
    n_layers: int = 12

    @classmethod
    def small(cls):
        """Small config for faster training/testing (good for Mac)"""
        return cls(
            d_model=256,
            n_heads=4,
            n_layers=4,
            n_ctx=256,  # Shorter context for faster training
            d_head=64,
            d_mlp=1024,
            d_vocab=50257,  # Will be updated by tokenizer
        )

    @classmethod
    def medium(cls):
        """Medium config (between small and full)"""
        return cls(
            d_model=512,
            n_heads=8,
            n_layers=6,
            n_ctx=512,
            d_head=64,
            d_mlp=2048,
            d_vocab=50257,
        )
