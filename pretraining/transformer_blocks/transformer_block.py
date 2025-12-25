from torch import nn
from jaxtyping import Float
from torch import Tensor
from pretraining.attention.attention import AttentionWithEinops, AttentionWithoutEinops
from pretraining.mlp.mlp import create_mlp_layer
from pretraining.normalization.layernorm import create_norm_layer


class TransformerBlockWithEinops(nn.Module):
    def __init__(self, cfg, rope=None, alibi=None):
        super().__init__()
        self.cfg = cfg
        self.ln1 = create_norm_layer(cfg, use_einops=True)
        self.attn = AttentionWithEinops(cfg, rope=rope, alibi=alibi)
        self.ln2 = create_norm_layer(cfg, use_einops=True)
        self.mlp = create_mlp_layer(cfg, use_einops=True)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # residual: [batch, posn, d_model]

        # Pre-norm attention with residual connection
        # ln1(residual): [batch, posn, d_model]
        # attn(...): [batch, posn, d_model]
        # residual: [batch, posn, d_model]
        residual = residual + self.attn(self.ln1(residual))

        # Pre-norm MLP with residual connection
        # ln2(residual): [batch, posn, d_model]
        # mlp(...): [batch, posn, d_model]
        # residual: [batch, posn, d_model]
        residual = residual + self.mlp(self.ln2(residual))

        return residual


class TransformerBlockWithoutEinops(nn.Module):
    def __init__(self, cfg, rope=None, alibi=None):
        super().__init__()
        self.cfg = cfg
        self.ln1 = create_norm_layer(cfg, use_einops=False)
        self.attn = AttentionWithoutEinops(cfg, rope=rope, alibi=alibi)
        self.ln2 = create_norm_layer(cfg, use_einops=False)
        self.mlp = create_mlp_layer(cfg, use_einops=False)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # residual: [batch, posn, d_model]

        # Pre-norm attention with residual connection
        # ln1(residual): [batch, posn, d_model]
        # attn(...): [batch, posn, d_model]
        # residual: [batch, posn, d_model]
        residual = residual + self.attn(self.ln1(residual))

        # Pre-norm MLP with residual connection
        # ln2(residual): [batch, posn, d_model]
        # mlp(...): [batch, posn, d_model]
        # residual: [batch, posn, d_model]
        residual = residual + self.mlp(self.ln2(residual))

        return residual


def create_transformer_block(cfg, use_einops=True, rope=None, alibi=None):
    """Factory function to create transformer block"""
    if use_einops:
        return TransformerBlockWithEinops(cfg, rope=rope, alibi=alibi)
    return TransformerBlockWithoutEinops(cfg, rope=rope, alibi=alibi)
