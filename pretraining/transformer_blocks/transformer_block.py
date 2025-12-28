"""Transformer Block implementation.

This module implements a single transformer block, which combines:
- Layer Normalization (pre-norm)
- Multi-Head Self-Attention
- Layer Normalization (pre-norm)
- MLP / Feedforward Network

Design Decision: Pre-norm vs Post-norm
- Pre-norm (what we use): x + f(LN(x)) - more stable for deep networks
- Post-norm: LN(x + f(x)) - original transformer, less stable for deep networks

We use pre-norm because it allows training deeper networks without gradient issues.
The residual connections allow gradients to flow directly through, enabling
very deep networks (100+ layers).

Architecture:
    Input
      ↓
    LayerNorm → Attention → + (residual)
      ↓                      ↑
      └──────────────────────┘
      ↓
    LayerNorm → MLP → + (residual)
      ↓                ↑
      └────────────────┘
      ↓
    Output
"""

from torch import nn
from jaxtyping import Float
from torch import Tensor
from typing import Optional, Tuple
from pretraining.attention.attention import Attention
from pretraining.mlp.mlp import create_mlp_layer
from pretraining.normalization.layernorm import create_norm_layer
from pretraining.utils import extract_mlp_output_and_aux_loss


class TransformerBlock(nn.Module):
    """Single Transformer Block.

    Combines attention and MLP with pre-norm layer normalization and residual
    connections. This is the fundamental building block of transformer models.

    Design Decision: Why two LayerNorms?
    - One before attention: Stabilizes attention computation
    - One before MLP: Stabilizes MLP computation
    - Each sub-block gets normalized inputs, preventing activation explosion

    Design Decision: Why residual connections?
    - Allow gradients to flow directly through (identity path)
    - Enable training of very deep networks (100+ layers)
    - Model can learn identity function if transformation isn't needed
    """

    def __init__(self, cfg, rope=None, alibi=None, use_einops=True):
        """Initialize transformer block.

        Args:
            cfg: Model configuration
            rope: RoPE module (None for GPT, RoPE instance for LLaMA)
            alibi: ALiBi module (None for GPT/LLaMA, ALiBi instance for OLMo)
            use_einops: If True, use einops implementations, else PyTorch
        """
        super().__init__()
        self.cfg = cfg
        # Pre-norm layer normalization before attention
        self.ln1 = create_norm_layer(cfg, use_einops=use_einops)
        # Multi-head self-attention
        self.attn = Attention(cfg, rope=rope, alibi=alibi,
                              use_einops=use_einops)
        # Pre-norm layer normalization before MLP
        self.ln2 = create_norm_layer(cfg, use_einops=use_einops)
        # MLP / Feedforward network (may be MoE)
        self.mlp = create_mlp_layer(cfg, use_einops=use_einops)

    def _apply_attention_with_residual(
        self,
        residual: Float[Tensor, "batch posn d_model"],
        cache: Optional[tuple[Float[Tensor, "batch cache_len n_kv_heads d_head"], Float[Tensor, "batch cache_len n_kv_heads d_head"]]],
        start_pos: int
    ) -> tuple[Float[Tensor, "batch posn d_model"], tuple[Float[Tensor, "batch new_cache_len n_kv_heads d_head"], Float[Tensor, "batch new_cache_len n_kv_heads d_head"]]]:
        """Apply attention sub-block with pre-norm and residual connection.

        Formula: output = input + Attention(LN(input))
        Pre-norm: normalize before attention (more stable)
        Residual: add input to output (gradient flow)

        Args:
            residual: Input tensor [batch, posn, d_model]
            cache: Optional KV cache tuple for efficient inference
            start_pos: Starting position for RoPE (used with cache)

        Returns:
            Tuple of (residual, cache) where:
            - residual: [batch, posn, d_model] - output after attention + residual
            - cache: Updated KV cache tuple
        """
        # Pre-norm: normalize before attention
        attn_output, new_cache = self.attn(
            self.ln1(residual), cache=cache, start_pos=start_pos
        )
        # Residual connection: add input to output
        residual = residual + attn_output
        return residual, new_cache

    def _apply_mlp_with_residual(
        self,
        residual: Float[Tensor, "batch posn d_model"]
    ) -> tuple[Float[Tensor, "batch posn d_model"], Optional[Float[Tensor, ""]]]:
        """Apply MLP sub-block with pre-norm and residual connection.

        Formula: output = input + MLP(LN(input))
        Pre-norm: normalize before MLP (more stable)
        Residual: add input to output (gradient flow)
        MLP may return (output, aux_loss) if MoE is enabled

        Args:
            residual: Input tensor [batch, posn, d_model]

        Returns:
            Tuple of (residual, aux_loss) where:
            - residual: [batch, posn, d_model] - output after MLP + residual
            - aux_loss: MoE auxiliary loss (None if not MoE)
        """
        # Pre-norm: normalize before MLP
        mlp_output = self.mlp(self.ln2(residual))
        # Extract output and aux_loss (handles both MoE and standard MLPs)
        mlp_output, aux_loss = extract_mlp_output_and_aux_loss(mlp_output)
        # Residual connection: add input to output
        residual = residual + mlp_output
        return residual, aux_loss

    def forward(
        self,
        residual: Float[Tensor, "batch posn d_model"],
        cache: Optional[tuple[Float[Tensor, "batch cache_len n_kv_heads d_head"],
                              Float[Tensor, "batch cache_len n_kv_heads d_head"]]] = None,
        start_pos: int = 0
    ) -> Tuple[Float[Tensor, "batch posn d_model"], tuple[Float[Tensor, "batch new_cache_len n_kv_heads d_head"], Float[Tensor, "batch new_cache_len n_kv_heads d_head"]], Optional[Float[Tensor, ""]]]:
        """Forward pass through transformer block.

        Args:
            residual: Input tensor [batch, posn, d_model]
            cache: Optional KV cache tuple for efficient inference
            start_pos: Starting position for RoPE (used with cache)

        Returns:
            Tuple of (output, (K_cache, V_cache), aux_loss) where:
            - output: [batch, posn, d_model] - block output
            - K_cache, V_cache: Updated KV cache
            - aux_loss: MoE auxiliary loss (None if not MoE)
        """
        # Step 1: Pre-norm attention with residual connection
        # Formula: output = input + Attention(LN(input))
        residual, new_cache = self._apply_attention_with_residual(
            residual, cache, start_pos
        )

        # Step 2: Pre-norm MLP with residual connection
        # Formula: output = input + MLP(LN(input))
        residual, aux_loss = self._apply_mlp_with_residual(residual)

        return residual, new_cache, aux_loss


# Backward compatibility aliases
TransformerBlockWithEinops = TransformerBlock
TransformerBlockWithoutEinops = TransformerBlock


def create_transformer_block(cfg, use_einops=True, rope=None, alibi=None):
    """Factory function to create transformer block.

    Args:
        cfg: Model configuration
        use_einops: If True, use einops implementations, else PyTorch
        rope: RoPE module (None for GPT, RoPE instance for LLaMA)
        alibi: ALiBi module (None for GPT/LLaMA, ALiBi instance for OLMo)

    Returns:
        TransformerBlock instance
    """
    return TransformerBlock(cfg, rope=rope, alibi=alibi, use_einops=use_einops)
