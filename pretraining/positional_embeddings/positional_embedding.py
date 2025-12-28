"""Learned Positional Embeddings (GPT style).

This module implements learned positional embeddings where each position gets
a learnable embedding vector. These embeddings are added to token embeddings
to provide position information to the model.

Design Decision: Learned vs Fixed Positional Encodings
- Learned (GPT): Each position has a learnable embedding - flexible, can adapt
- RoPE (LLaMA): Rotates Q/K vectors by position-dependent angles - better extrapolation
- ALiBi (OLMo): Adds distance-based bias to attention scores - no learned params

We use learned embeddings for GPT-style models because they're simple and work well
for the context lengths we train on.

Mathematical Formula:
    final_embedding = token_embedding + positional_embedding
    
Where:
    - token_embedding: [batch, seq_len, d_model] - from token IDs
    - positional_embedding: [batch, seq_len, d_model] - from position IDs
    - Positional embedding is learned: W_pos[i] is embedding for position i
"""

import torch
import torch.nn as nn
import einops
from jaxtyping import Float, Int
from torch import Tensor


class PosEmbed(nn.Module):
    """Learned Positional Embeddings (GPT style).
    
    Each position in the sequence gets a learnable embedding vector. These are
    added to token embeddings to provide position information.
    
    Design Decision: Why learned embeddings?
    - Simple: Just a lookup table W_pos[position]
    - Flexible: Model can learn optimal position representations
    - Works well: GPT-2, GPT-3 use this approach
    - Limitation: Can't extrapolate beyond max context length (n_ctx)
    """
    
    def __init__(self, cfg, use_einops=True):
        """Initialize positional embedding layer.
        
        Args:
            cfg: Model configuration
            use_einops: If True, use einops for tensor operations (more explicit),
                       else use PyTorch operations (more standard)
        """
        super().__init__()
        self.cfg = cfg
        self.use_einops = use_einops
        # W_pos: [n_ctx, d_model] - positional embedding matrix
        # W_pos[i] is the embedding for position i
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def _broadcast_embeddings(
        self, 
        position_embeddings: Float[Tensor, "seq_len d_model"],
        batch_size: int
    ) -> Float[Tensor, "batch seq_len d_model"]:
        """Broadcast positional embeddings to batch dimension.
        
        Args:
            position_embeddings: Embeddings [seq_len, d_model]
            batch_size: Batch size
        
        Returns:
            Broadcasted embeddings [batch, seq_len, d_model]
        """
        if self.use_einops:
            return einops.repeat(
                position_embeddings, "seq d_model -> batch seq d_model", batch=batch_size
            )
        else:
            # Manual broadcasting:
            # 1. Add batch dimension: [seq_len, d_model] -> [1, seq_len, d_model]
            position_embeddings_with_batch_dim = position_embeddings.unsqueeze(0)
            # 2. Expand along batch dimension (memory-efficient, no copying)
            return position_embeddings_with_batch_dim.expand(batch_size, -1, -1)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        """Forward pass through positional embedding layer.
        
        Args:
            tokens: Token IDs [batch, position] (used to get sequence length)
        
        Returns:
            Positional embeddings [batch, position, d_model]
        """
        batch, seq_len = tokens.shape
        
        # Get embeddings for current sequence length
        # W_pos[:seq_len]: [seq_len, d_model] - embeddings for positions 0 to seq_len-1
        position_embeddings = self.W_pos[:seq_len]
        
        # Broadcast to batch dimension
        return self._broadcast_embeddings(position_embeddings, batch)


# Backward compatibility aliases
PosEmbedWithEinops = PosEmbed
PosEmbedWithoutEinops = PosEmbed
