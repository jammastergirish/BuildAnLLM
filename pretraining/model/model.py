"""Full Transformer Model implementation.

This module implements the complete transformer language model, combining:
- Token Embeddings: Convert token IDs to dense vectors
- Positional Embeddings: Add position information (GPT style) or use RoPE/ALiBi
- Transformer Blocks: Stack of attention + MLP blocks
- Final Layer Normalization
- Unembedding: Convert hidden states to vocabulary logits

The model supports multiple architectures:
- GPT: Learned positional embeddings, LayerNorm, GELU
- LLaMA: RoPE positional encoding, RMSNorm, SwiGLU
- OLMo: ALiBi positional encoding, LayerNorm, SwiGLU

Design Decision: Why different architectures?
- GPT: Original transformer, simple and effective
- LLaMA: Better extrapolation (RoPE), simpler normalization (RMSNorm)
- OLMo: Better long-context handling (ALiBi), simpler than RoPE

Mathematical Flow:
    tokens → embeddings → + positional → transformer blocks → LN → unembedding → logits
"""

from __future__ import annotations

import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from typing import Optional, Union, Tuple
from config import Architecture, PositionalEncoding
from pretraining.embeddings.embed import EmbedWithoutTorch, EmbedWithTorch, UnembedWithoutTorch, UnembedWithTorch
from pretraining.positional_embeddings.positional_embedding import PosEmbed
from pretraining.transformer_blocks.transformer_block import create_transformer_block
from pretraining.normalization.layernorm import create_norm_layer
from pretraining.positional_embeddings.rope import RoPE
from pretraining.utils import extract_model_output_and_aux_loss


def _aggregate_aux_losses(aux_losses: list) -> Optional[Float[Tensor, ""]]:
    """Aggregate auxiliary losses from multiple MoE layers.

    When using MoE, each transformer block's MLP can return an auxiliary
    load balancing loss. This function sums them all together.

    Args:
        aux_losses: List of auxiliary losses (may contain None values)

    Returns:
        Sum of all auxiliary losses, or None if no losses
    """
    if aux_losses:
        return sum(aux_losses)
    return None


# Type alias for forward method return type (using string to avoid parsing issues)
ForwardReturnType: str = "Union[Float[Tensor, 'batch position d_vocab'], Tuple[Float[Tensor, 'batch position d_vocab'], Optional[Float[Tensor, \"\"]]], Tuple[Float[Tensor, 'batch position d_vocab'], Optional[list[tuple[Float[Tensor, \"batch new_cache_len n_heads d_head\"], Float[Tensor, \"batch new_cache_len n_heads d_head\"]]]], Tuple[Float[Tensor, 'batch position d_vocab'], Optional[list[tuple[Float[Tensor, \"batch new_cache_len n_heads d_head\"], Float[Tensor, \"batch new_cache_len n_heads d_head\"]]], Optional[Float[Tensor, \"\"]]]]"


class TransformerModel(nn.Module):
    """Complete Transformer Language Model.

    This is the full model that combines all components into a language model
    capable of autoregressive text generation. It supports GPT, LLaMA, and OLMo
    architectures through configuration.

    Architecture Flow:
        Tokens [batch, position]
          ↓
        Token Embeddings → [batch, position, d_model]
          ↓
        + Positional Embeddings (GPT) or RoPE/ALiBi in attention (LLaMA/OLMo)
          ↓
        Transformer Block 1 → [batch, position, d_model]
          ↓
        ...
          ↓
        Transformer Block N → [batch, position, d_model]
          ↓
        Final LayerNorm → [batch, position, d_model]
          ↓
        Unembedding → [batch, position, d_vocab] (logits)

    Design Decision: Why final LayerNorm?
    - Normalizes activations before converting to logits
    - Stabilizes training
    - Standard practice in transformer models
    """

    def __init__(self, cfg, use_einops=True):
        """Initialize transformer model.

        Args:
            cfg: Model configuration
            use_einops: If True, use einops implementations, else PyTorch
        """
        super().__init__()
        self.cfg = cfg
        self.use_einops = use_einops

        # Token embeddings: Convert token IDs to dense vectors
        # Same for all architectures
        if use_einops:
            self.embed = EmbedWithoutTorch(cfg)
        else:
            self.embed = EmbedWithTorch(cfg)

        # Positional embeddings (GPT style: learned embeddings)
        # LLaMA uses RoPE (applied in attention), OLMo uses ALiBi (applied in attention)
        if cfg.positional_encoding == PositionalEncoding.LEARNED:
            self.pos_embed = PosEmbed(cfg, use_einops=use_einops)
        else:
            self.pos_embed = None

        # RoPE (Rotary Position Embedding) for LLaMA-style models
        # Applied to Q/K in attention, not added to embeddings
        if cfg.positional_encoding == PositionalEncoding.ROPE:
            self.rope = RoPE(cfg)
        else:
            self.rope = None

        # ALiBi (Attention with Linear Biases) for OLMo-style models
        # Applied to attention scores, not added to embeddings
        if cfg.positional_encoding == PositionalEncoding.ALIBI:
            from pretraining.positional_embeddings.alibi import ALiBi
            self.alibi = ALiBi(cfg)
        else:
            self.alibi = None

        # Transformer blocks: Stack of attention + MLP blocks
        # Each block processes the sequence and passes it to the next
        self.blocks = nn.ModuleList([
            create_transformer_block(
                cfg, use_einops=use_einops, rope=self.rope, alibi=self.alibi)
            for _ in range(cfg.n_layers)
        ])

        # Final normalization: Normalize before converting to logits
        self.ln_f = create_norm_layer(cfg, use_einops=use_einops)

        # Unembedding: Convert hidden states to vocabulary logits
        # Projects from d_model to d_vocab
        if use_einops:
            self.unembed = UnembedWithoutTorch(cfg)
        else:
            self.unembed = UnembedWithTorch(cfg)

    def _process_blocks(
        self,
        residual: Float[Tensor, "batch position d_model"],
        cache: Optional[list[tuple[Float[Tensor, "batch cache_len n_kv_heads d_head"],
                                   Float[Tensor, "batch cache_len n_kv_heads d_head"]]]],
        start_pos: int
    ) -> Tuple[Float[Tensor, "batch position d_model"], list, list]:
        """Process input through all transformer blocks.

        Iterates through transformer blocks, handling cache management and
        collecting auxiliary losses from MoE layers.

        Args:
            residual: Input tensor [batch, position, d_model]
            cache: Optional KV cache list (one per layer) for efficient inference
            start_pos: Starting position for RoPE (used with cache)

        Returns:
            Tuple of (residual, new_cache_list, aux_losses) where:
            - residual: [batch, position, d_model] - output after all blocks
            - new_cache_list: List of updated KV caches (one per block)
            - aux_losses: List of auxiliary losses from MoE layers
        """
        new_cache_list = []
        aux_losses = []

        for i, block in enumerate(self.blocks):
            # Get cache for this layer (if provided)
            block_cache = cache[i] if cache is not None else None
            # Forward through block
            residual, new_cache, aux_loss = block(
                residual, cache=block_cache, start_pos=start_pos)
            # Store updated cache
            new_cache_list.append(new_cache)
            # Collect auxiliary losses (from MoE layers)
            if aux_loss is not None:
                aux_losses.append(aux_loss)

        return residual, new_cache_list, aux_losses

    def _format_output(
        self,
        logits: Float[Tensor, "batch position d_vocab"],
        cache: Optional[list],
        aux_loss: Optional[Float[Tensor, ""]]
    ) -> ForwardReturnType:
        """Format model output based on cache and aux_loss presence.

        Handles different return formats to maintain backward compatibility:
        - Standard: logits only
        - With cache: (logits, cache)
        - With aux_loss: (logits, aux_loss)
        - With both: (logits, cache, aux_loss)

        Args:
            logits: Model logits [batch, position, d_vocab]
            cache: Optional KV cache list (one per layer)
            aux_loss: Optional auxiliary loss from MoE layers

        Returns:
            Formatted output (logits, tuple, or combination)
        """
        if cache is not None:
            if aux_loss is not None:
                return logits, cache, aux_loss
            else:
                return logits, cache
        else:
            if aux_loss is not None:
                return logits, aux_loss
            else:
                return logits

    def forward(
        self,
        tokens: Int[Tensor, "batch position"],
        cache: Optional[list[tuple[Float[Tensor, "batch cache_len n_kv_heads d_head"],
                                   Float[Tensor, "batch cache_len n_kv_heads d_head"]]]] = None,
        start_pos: int = 0
    ) -> ForwardReturnType:
        """Forward pass through transformer model.

        Args:
            tokens: Token IDs [batch, position]
            cache: Optional KV cache list (one per layer) for efficient inference
            start_pos: Starting position for RoPE (used with cache)

        Returns:
            Logits [batch, position, d_vocab] or tuple with cache/aux_loss
        """
        # tokens: [batch, position]

        # Step 1: Token embeddings
        # Convert token IDs to dense vectors
        # residual: [batch, position, d_model]
        residual = self.embed(tokens)

        # Step 2: Positional embeddings (GPT style only)
        # LLaMA uses RoPE (applied in attention), OLMo uses ALiBi (applied in attention)
        if self.pos_embed is not None:
            # pos_emb: [batch, position, d_model]
            # residual: [batch, position, d_model]
            # Add positional embeddings to token embeddings
            residual = residual + self.pos_embed(tokens)

        # Step 3: Pass through transformer blocks
        # Each block: [batch, position, d_model] -> [batch, position, d_model]
        # Blocks process the sequence through attention and MLP
        residual, new_cache_list, aux_losses = self._process_blocks(
            residual, cache, start_pos)

        # Step 4: Aggregate auxiliary losses from all MoE layers
        # If multiple MoE layers exist, sum their load balancing losses
        total_aux_loss = _aggregate_aux_losses(aux_losses)

        # Step 5: Final layer normalization
        # Normalize activations before converting to logits
        # residual: [batch, position, d_model]
        residual = self.ln_f(residual)

        # Step 6: Unembedding to logits
        # Project from d_model to d_vocab (vocabulary size)
        # residual: [batch, position, d_model]
        # logits: [batch, position, d_vocab]
        logits = self.unembed(residual)

        # Step 7: Format output (maintain backward compatibility)
        return self._format_output(logits, new_cache_list if cache is not None else None, total_aux_loss)


# Backward compatibility aliases
TransformerModelWithEinops = TransformerModel
TransformerModelWithoutEinops = TransformerModel
GPTWithEinops = TransformerModel
GPTWithoutEinops = TransformerModel
