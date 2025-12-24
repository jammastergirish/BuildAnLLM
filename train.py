# /// script
# dependencies = ["torch, einops"]
# ///

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from dataclasses import dataclass
from jaxtyping import Float, Tensor
from layernorm import LayerNormWithEinops, LayerNormWithoutEinops, LayerNormWithTorch
from embed import EmbedWithoutTorch, EmbedWithTorch

device = torch.device(
    "mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu"
)


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
