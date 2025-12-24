# /// script
# dependencies = ["torch, einops"]
# ///

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from jaxtyping import Float, Tensor
from config import GPTConfig
from layernorm import LayerNormWithEinops, LayerNormWithoutEinops, LayerNormWithTorch
from embed import EmbedWithoutTorch, EmbedWithTorch
from positional_embedding import PosEmbedWithEinops, PosEmbedWithoutEinops
from attention import AttentionWithEinops, AttentionWithoutEinops
from mlp import MLPWithEinops, MLPWithoutEinops
from transformer_block import TransformerBlockWithEinops, TransformerBlockWithoutEinops
from gpt import GPTWithEinops, GPTWithoutEinops

device = torch.device(
    "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"
)
