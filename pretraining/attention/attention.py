"""Multi-Head Self-Attention with support for GPT, LLaMA, and OLMo architectures.

This module implements scaled dot-product attention with support for:
- Multi-Head Attention (MHA): Standard attention with separate Q, K, V per head
- Grouped Query Attention (GQA): Multiple Q heads share K/V heads (memory efficient)
- Multi-Query Attention (MQA): All Q heads share single K/V head (most efficient)
- RoPE (Rotary Position Embedding): Applied to Q/K in LLaMA-style models
- ALiBi (Attention with Linear Biases): Applied to attention scores in OLMo-style models
- KV Caching: Efficient inference by caching K/V tensors

Design Decision: We provide both einops and PyTorch implementations to show
different approaches. Einops makes tensor operations explicit and readable,
while PyTorch operations are more standard and potentially faster.
"""

import torch
import torch.nn as nn
import einops
from jaxtyping import Float
from torch import Tensor
from typing import Optional, Union


class Attention(nn.Module):
    """Multi-Head Self-Attention supporting MHA, GQA, and MQA.
    
    This class implements scaled dot-product attention with optional RoPE and ALiBi.
    It supports both einops (explicit, readable) and PyTorch (standard) implementations.
    
    Design Decisions:
    - Pre-norm architecture: Normalize before attention (more stable for deep networks)
    - KV caching: Cache K/V but not Q (Q changes each step, K/V don't)
    - GQA/MQA: Use fewer KV heads than Q heads to save memory (75% reduction for 4:1 ratio)
    - Causal masking: Prevent attending to future tokens (essential for autoregressive models)
    
    Mathematical Formula:
        Attention(Q, K, V) = softmax(QK^T / √d_head + mask + bias) @ V
        
    Where:
        - Q, K, V: Query, Key, Value projections
        - √d_head: Scaling factor to prevent softmax saturation
        - mask: Causal mask (-inf for future positions)
        - bias: ALiBi bias (distance-based, per-head slopes)
    """
    
    def __init__(self, cfg, rope=None, alibi=None, use_einops=True):
        """Initialize attention layer.
        
        Args:
            cfg: Model configuration
            rope: RoPE module (None for GPT, RoPE instance for LLaMA)
            alibi: ALiBi module (None for GPT/LLaMA, ALiBi instance for OLMo)
            use_einops: If True, use einops for tensor operations (more explicit),
                       else use PyTorch operations (more standard)
        """
        super().__init__()
        self.cfg = cfg
        self.use_einops = use_einops
        self.rope = rope  # RoPE module (None for GPT, RoPE instance for LLaMA)
        self.alibi = alibi  # ALiBi module (None for GPT/LLaMA, ALiBi instance for OLMo)
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads  # Number of KV heads (for GQA/MQA)
        
        # W_Q: [n_heads, d_head, d_model] - separate Q projection for each head
        self.W_Q = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))
        # W_K, W_V: [n_kv_heads, d_head, d_model] - shared K/V heads for GQA/MQA
        # For MHA: n_kv_heads = n_heads (standard)
        # For GQA: n_kv_heads < n_heads (e.g., 8 KV heads for 32 Q heads = 4:1 ratio)
        # For MQA: n_kv_heads = 1 (all Q heads share single K/V head)
        self.W_K = nn.Parameter(torch.empty((cfg.n_kv_heads, cfg.d_head, cfg.d_model)))
        self.W_V = nn.Parameter(torch.empty((cfg.n_kv_heads, cfg.d_head, cfg.d_model)))
        # W_O: [n_heads, d_head, d_model] - output projection (combines all heads)
        self.W_O = nn.Parameter(torch.empty((cfg.n_heads, cfg.d_head, cfg.d_model)))

        for param in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.normal_(param, std=self.cfg.init_range)

    def _compute_qkv(self, residual: Float[Tensor, "batch posn d_model"]) -> tuple[Float[Tensor, "batch posn n_heads d_head"], Float[Tensor, "batch posn n_kv_heads d_head"], Float[Tensor, "batch posn n_kv_heads d_head"]]:
        """Compute Q, K, V projections.
        
        Args:
            residual: Input tensor [batch, posn, d_model]
        
        Returns:
            Tuple of (q, k, v) where:
            - q: [batch, posn, n_heads, d_head]
            - k: [batch, posn, n_kv_heads, d_head]
            - v: [batch, posn, n_kv_heads, d_head]
        """
        if self.use_einops:
            q = einops.einsum(
                residual, self.W_Q,
                "batch posn d_model, n_heads d_head d_model -> batch posn n_heads d_head"
            )
            k = einops.einsum(
                residual, self.W_K,
                "batch posn d_model, n_kv_heads d_head d_model -> batch posn n_kv_heads d_head"
            )
            v = einops.einsum(
                residual, self.W_V,
                "batch posn d_model, n_kv_heads d_head d_model -> batch posn n_kv_heads d_head"
            )
        else:
            q = torch.einsum("bpd,nhd->bpnh", residual, self.W_Q)
            k = torch.einsum("bpd,nkd->bpnk", residual, self.W_K)
            v = torch.einsum("bpd,nkd->bpnk", residual, self.W_V)
        return q, k, v

    def _compute_attention_scores(
        self, 
        q: Float[Tensor, "batch posn_q n_heads d_head"],
        k: Float[Tensor, "batch posn_k n_heads d_head"]
    ) -> Float[Tensor, "batch n_heads posn_q posn_k"]:
        """Compute scaled dot-product attention scores.
        
        Formula: attn_scores = Q @ K^T / √d_head
        Why scale by √d_head?
        - Dot products grow large as d_head increases
        - Large values → softmax saturates → gradients vanish
        - Scaling prevents this: keeps values in reasonable range
        - This is the "scaled" in "scaled dot-product attention"
        
        Args:
            q: Query tensor [batch, posn_q, n_heads, d_head]
            k: Key tensor [batch, posn_k, n_heads, d_head] (may be broadcasted)
        
        Returns:
            Attention scores [batch, n_heads, posn_q, posn_k]
        """
        if self.use_einops:
            return einops.einsum(
                q, k,
                "batch posn_q n_heads d_head, batch posn_k n_heads d_head -> batch n_heads posn_q posn_k"
            ) / (self.cfg.d_head ** 0.5)
        else:
            # Transpose to [batch, n_heads, seq_len, d_head] for matmul
            q_t = q.transpose(1, 2)  # [batch, n_heads, posn_q, d_head]
            k_t = k.transpose(1, 2)  # [batch, n_heads, posn_k, d_head]
            return torch.matmul(q_t, k_t.transpose(-2, -1)) / (self.cfg.d_head ** 0.5)

    def _apply_attention_to_values(
        self,
        attn_pattern: Float[Tensor, "batch n_heads posn_q posn_k"],
        v: Float[Tensor, "batch posn_k n_heads d_head"]
    ) -> Float[Tensor, "batch posn_q n_heads d_head"]:
        """Apply attention pattern to values.
        
        Formula: output = attn_pattern @ V
        Weighted sum of values: output[i] = Σ_j attn_pattern[i, j] * V[j]
        
        Args:
            attn_pattern: Attention probabilities [batch, n_heads, posn_q, posn_k]
            v: Value tensor [batch, posn_k, n_heads, d_head]
        
        Returns:
            Attention output [batch, posn_q, n_heads, d_head]
        """
        if self.use_einops:
            return einops.einsum(
                attn_pattern, v,
                "batch n_heads posn_q posn_k, batch posn_k n_heads d_head -> batch posn_q n_heads d_head"
            )
        else:
            # v needs to be transposed for matmul: [batch, n_heads, posn_k, d_head]
            v_t = v.transpose(1, 2)
            attn_output = torch.matmul(attn_pattern, v_t)  # [batch, n_heads, posn_q, d_head]
            return attn_output.transpose(1, 2)  # [batch, posn_q, n_heads, d_head]

    def _project_output(
        self,
        attn_output: Float[Tensor, "batch posn n_heads d_head"]
    ) -> Float[Tensor, "batch posn d_model"]:
        """Project attention output back to d_model.
        
        Combines all heads and projects to model dimension.
        
        Args:
            attn_output: Attention output [batch, posn, n_heads, d_head]
        
        Returns:
            Projected output [batch, posn, d_model]
        """
        if self.use_einops:
            return einops.einsum(
                attn_output, self.W_O,
                "batch posn n_heads d_head, n_heads d_head d_model -> batch posn d_model"
            )
        else:
            return torch.einsum("bpnh,nhd->bpd", attn_output, self.W_O)

    def forward(
        self, 
        residual: Float[Tensor, "batch posn d_model"],
        cache: Optional[tuple[Float[Tensor, "batch cache_len n_kv_heads d_head"], Float[Tensor, "batch cache_len n_kv_heads d_head"]]] = None,
        start_pos: int = 0,
        return_attention_pattern: bool = False
    ) -> Union[
        tuple[Float[Tensor, "batch posn d_model"], tuple[Float[Tensor, "batch new_cache_len n_kv_heads d_head"], Float[Tensor, "batch new_cache_len n_kv_heads d_head"]]],
        tuple[Float[Tensor, "batch posn d_model"], tuple[Float[Tensor, "batch new_cache_len n_kv_heads d_head"], Float[Tensor, "batch new_cache_len n_kv_heads d_head"]], Float[Tensor, "batch n_heads posn_q posn_k"]]
    ]:
        """Forward pass through attention layer.
        
        Args:
            residual: Input tensor [batch, posn, d_model]
            cache: Optional KV cache tuple (K_cache, V_cache) for efficient inference
            start_pos: Starting position for RoPE (used with cache)
            return_attention_pattern: If True, return attention weights
        
        Returns:
            Tuple of (output, (K_cache, V_cache)) where:
            - output: [batch, posn, d_model] - attention output
            - K_cache, V_cache: [batch, new_cache_len, n_kv_heads, d_head] - updated cache
            
            If return_attention_pattern is True, returns (output, cache, attn_pattern)
        """
        seq_len = residual.shape[1]

        # Step 1: Compute Q, K, V projections
        q, k, v = self._compute_qkv(residual)

        # Step 2: Handle KV cache for efficient inference
        # Why cache K/V but not Q?
        # - During autoregressive generation, we generate tokens one at a time
        # - Q changes for each new token (we're querying with the new token)
        # - K/V for previous tokens don't change (they're already computed)
        # - By caching K/V, we avoid recomputing them each step (10-100x speedup)
        # Note: Cached K, V already have RoPE applied (if using RoPE)
        # Cache shape: [batch, cache_len, n_kv_heads, d_head] (smaller than MHA for GQA/MQA!)
        if cache is not None:
            # Handle cache format: should be tuple (k_cache, v_cache) or list [k_cache, v_cache]
            if isinstance(cache, (list, tuple)) and len(cache) == 2:
                k_cache, v_cache = cache
            else:
                raise ValueError(f"Cache must be tuple or list of 2 elements, got {type(cache)} with length {len(cache) if hasattr(cache, '__len__') else 'N/A'}")
            k = torch.cat([k_cache, k], dim=1)  # [batch, cache_len + seq_len, n_kv_heads, d_head]
            v = torch.cat([v_cache, v], dim=1)  # [batch, cache_len + seq_len, n_kv_heads, d_head]
            total_len = k.shape[1]
        else:
            total_len = seq_len

        # Step 3: Apply RoPE (Rotary Position Embedding) if provided (LLaMA-style)
        # RoPE encodes position by rotating Q/K vectors by position-dependent angles
        # For cached K, RoPE was already applied, so we only apply to new positions
        # Note: RoPE is applied to K with n_kv_heads shape (before broadcasting)
        if self.rope is not None:
            positions = torch.arange(start_pos, start_pos + seq_len, device=residual.device)
            q, k_new = self.rope(q, k[:, -seq_len:, :, :], positions)
            if cache is not None:
                k = torch.cat([k[:, :-seq_len, :, :], k_new], dim=1)
            else:
                k = k_new

        # Step 4: Store original K/V for caching (before broadcasting)
        # We cache the original n_kv_heads version to save memory
        # After broadcasting, K/V would be [batch, seq_len, n_heads, d_head]
        # But we only need to cache [batch, seq_len, n_kv_heads, d_head] (smaller!)
        # If cache was provided, k/v already contains accumulated sequence, so cache the entire thing
        # Otherwise, cache only the new tokens
        if cache is not None:
            k_for_cache = k  # [batch, total_len, n_kv_heads, d_head] - accumulated sequence
            v_for_cache = v  # [batch, total_len, n_kv_heads, d_head] - accumulated sequence
        else:
            k_for_cache = k[:, -seq_len:, :, :]  # [batch, seq_len, n_kv_heads, d_head]
            v_for_cache = v[:, -seq_len:, :, :]  # [batch, seq_len, n_kv_heads, d_head]

        # Step 5: Broadcast K/V to match Q heads for GQA/MQA
        # For MHA: n_kv_heads = n_heads, so no broadcasting needed
        # For GQA/MQA: n_kv_heads < n_heads, so we repeat each KV head
        # Example: 32 Q heads, 8 KV heads → each KV head serves 4 Q heads
        # This saves memory: cache is 75% smaller for 4:1 ratio
        if self.n_kv_heads < self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)  # [batch, posn_k, n_heads, d_head]
            v = v.repeat_interleave(repeat_factor, dim=2)  # [batch, posn_k, n_heads, d_head]

        # Step 6: Compute scaled dot-product attention scores
        attn_scores = self._compute_attention_scores(q, k)

        # Step 7: Apply ALiBi bias if provided (OLMo-style)
        # ALiBi adds distance-based bias to attention scores
        # Formula: bias[h, i, j] = -slope[h] * |i - j| for future positions
        # Each head gets different slope: slope[h] = 2^(-8/n_heads * h)
        # Closer positions get less negative bias (can attend more)
        # Farther positions get more negative bias (attend less)
        if self.alibi is not None:
            alibi_bias = self.alibi.get_bias(total_len, residual.device)  # [n_heads, total_len, total_len]
            attn_scores = attn_scores + alibi_bias.unsqueeze(0)[:, :, start_pos:start_pos+seq_len, :]

        # Step 8: Apply causal mask
        # Why causal masking?
        # - During inference, we generate tokens one at a time (autoregressive)
        # - Model should only use past context, not future tokens
        # - Training must match inference conditions
        # - Mask prevents attending to future positions
        # mask: [seq_len, total_len] - lower triangular matrix
        # mask[i, j] = 1 if j <= i (can attend), else 0 (cannot attend)
        mask = torch.tril(torch.ones((seq_len, total_len), device=residual.device))
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Step 9: Apply softmax to get attention probabilities
        # Formula: attn_pattern = softmax(attn_scores)
        # Softmax converts scores to probabilities (sum to 1 over keys)
        # attn_pattern[b, h, i, j] = probability that position i attends to position j in head h
        attn_pattern = torch.softmax(attn_scores, dim=-1)

        # Step 10: Apply attention to values
        attn_output = self._apply_attention_to_values(attn_pattern, v)

        # Step 11: Project back to d_model
        output = self._project_output(attn_output)

        # Return cache: use the original (non-broadcasted) K/V to save memory
        if return_attention_pattern:
            return output, (k_for_cache, v_for_cache), attn_pattern
        return output, (k_for_cache, v_for_cache)


# Backward compatibility aliases
AttentionWithEinops = Attention
AttentionWithoutEinops = Attention
