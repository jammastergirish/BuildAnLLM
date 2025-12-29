"""Utilities for applying LoRA to transformer models."""

import torch
import torch.nn as nn
import einops
from typing import List, Optional
from finetuning.peft.lora_wrappers import (
    apply_lora_to_attention,
    apply_lora_to_mlp,
    einsum_with_lora,
)


def convert_model_to_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    dropout: float = 0.0,
    target_modules: Optional[str] = None,
) -> nn.Module:
    """
    Convert a transformer model to use LoRA adapters.
    
    Args:
        model: The transformer model to convert
        rank: LoRA rank (r) - dimension of low-rank matrices
        alpha: LoRA alpha - scaling factor (typically = rank)
        dropout: Optional dropout for LoRA adapters
        target_modules: Which modules to apply LoRA to. Options:
            - "all": Apply to all linear layers (attention + MLP)
            - "attention": Only attention layers (W_Q, W_K, W_V, W_O)
            - "mlp": Only MLP layers (W_in, W_out)
    
    Returns:
        Model with LoRA adapters applied
    """
    if target_modules is None:
        target_modules = "all"
    
    # Freeze all base parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Apply LoRA to transformer blocks
    lora_count = 0
    if hasattr(model, 'blocks'):
        for block in model.blocks:
            # Apply to attention layer
            if hasattr(block, 'attn') and (target_modules == "all" or target_modules == "attention"):
                apply_lora_to_attention(block.attn, rank, alpha, dropout)
                _patch_attention_forward(block.attn)
                lora_count += 4  # W_Q, W_K, W_V, W_O
            
            # Apply to MLP layer
            if hasattr(block, 'mlp') and (target_modules == "all" or target_modules == "mlp"):
                apply_lora_to_mlp(block.mlp, rank, alpha, dropout)
                _patch_mlp_forward(block.mlp)
                lora_count += 2  # W_in, W_out
    
    print(f"Applied LoRA to {lora_count} parameter matrices")
    print(f"LoRA rank: {rank}, alpha: {alpha}, target: {target_modules}")
    
    return model


def _patch_attention_forward(attn_module):
    """
    Patch attention forward method to use LoRA.
    
    This dynamically replaces the forward method with a LoRA-aware version.
    Works for both einops and non-einops attention implementations.
    
    Args:
        attn_module: Attention module to patch (AttentionWithEinops or AttentionWithoutEinops)
    """
    # Check if using einops by checking the class name
    use_einops = 'WithEinops' in attn_module.__class__.__name__
    
    def lora_forward(self, residual, cache=None, start_pos=0, return_attention_pattern=False, **kwargs):
        """
        Forward pass with LoRA support.
        
        Computes Q, K, V, and output projection with LoRA adapters if enabled.
        Maintains compatibility with RoPE and ALiBi positional encodings.
        
        Args:
            residual: Input tensor [batch, posn, d_model]
            cache: Optional KV cache tuple (K_cache, V_cache) for efficient inference
            start_pos: Starting position for RoPE (used with cache)
            return_attention_pattern: If True, return attention weights
        
        Returns:
            Tuple of (output, (K_cache, V_cache)) or (output, (K_cache, V_cache), attn_pattern)
        """
        seq_len = residual.shape[1]
        
        # Compute Q, K, V with LoRA
        def compute_with_lora(x, weight, pattern, param_name):
            """Compute einsum with LoRA if enabled, otherwise standard einsum."""
            if hasattr(self, f'{param_name}_use_lora'):
                return einsum_with_lora(
                    x, weight, pattern,
                    getattr(self, f'{param_name}_lora_A'),
                    getattr(self, f'{param_name}_lora_B'),
                    getattr(self, f'{param_name}_lora_scaling'),
                    getattr(self, f'{param_name}_lora_dropout'),
                )
            else:
                # Standard einsum (einops works for both einops and non-einops models)
                return einops.einsum(x, weight, pattern)
        
        # Step 1: Compute Q, K, V projections with LoRA
        # Note: For GQA/MQA, K/V have n_kv_heads, but we compute with n_heads first
        # We'll handle broadcasting later
        q = compute_with_lora(
            residual, self.W_Q,
            "batch posn d_model, n_heads d_head d_model -> batch posn n_heads d_head",
            "W_Q"
        )
        k = compute_with_lora(
            residual, self.W_K,
            "batch posn d_model, n_kv_heads d_head d_model -> batch posn n_kv_heads d_head",
            "W_K"
        )
        v = compute_with_lora(
            residual, self.W_V,
            "batch posn d_model, n_kv_heads d_head d_model -> batch posn n_kv_heads d_head",
            "W_V"
        )
        
        # Step 2: Handle KV cache
        if cache is not None:
            if isinstance(cache, (list, tuple)) and len(cache) == 2:
                k_cache, v_cache = cache
            else:
                raise ValueError(f"Cache must be tuple or list of 2 elements, got {type(cache)}")
            k = torch.cat([k_cache, k], dim=1)  # [batch, cache_len + seq_len, n_kv_heads, d_head]
            v = torch.cat([v_cache, v], dim=1)  # [batch, cache_len + seq_len, n_kv_heads, d_head]
            total_len = k.shape[1]
        else:
            total_len = seq_len
        
        # Step 3: Apply RoPE if provided
        if self.rope is not None:
            positions = torch.arange(start_pos, start_pos + seq_len, device=residual.device)
            q, k_new = self.rope(q, k[:, -seq_len:, :, :], positions)
            if cache is not None:
                k = torch.cat([k[:, :-seq_len, :, :], k_new], dim=1)
            else:
                k = k_new
        
        # Step 4: Store K/V for caching (before broadcasting)
        if cache is not None:
            k_for_cache = k  # [batch, total_len, n_kv_heads, d_head]
            v_for_cache = v  # [batch, total_len, n_kv_heads, d_head]
        else:
            k_for_cache = k[:, -seq_len:, :, :]  # [batch, seq_len, n_kv_heads, d_head]
            v_for_cache = v[:, -seq_len:, :, :]  # [batch, seq_len, n_kv_heads, d_head]
        
        # Step 5: Broadcast K/V to match Q heads for GQA/MQA
        if self.n_kv_heads < self.n_heads:
            repeat_factor = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=2)  # [batch, posn_k, n_heads, d_head]
            v = v.repeat_interleave(repeat_factor, dim=2)  # [batch, posn_k, n_heads, d_head]
        
        # Step 6: Compute attention scores
        if use_einops:
            attn_scores = einops.einsum(
                q, k,
                "batch posn_q n_heads d_head, batch posn_k n_heads d_head -> batch n_heads posn_q posn_k"
            ) / (self.cfg.d_head ** 0.5)
        else:
            q = q.transpose(1, 2)  # [batch, n_heads, posn, d_head]
            k = k.transpose(1, 2)  # [batch, n_heads, posn, d_head]
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.cfg.d_head ** 0.5)
        
        # Step 7: Apply ALiBi if provided
        if self.alibi is not None:
            alibi_bias = self.alibi.get_bias(total_len, residual.device)  # [n_heads, total_len, total_len]
            attn_scores = attn_scores + alibi_bias.unsqueeze(0)[:, :, start_pos:start_pos+seq_len, :]
        
        # Step 8: Apply causal mask
        mask = torch.tril(torch.ones((seq_len, total_len), device=residual.device))
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))
        
        # Step 9: Softmax
        attn_pattern = torch.softmax(attn_scores, dim=-1)
        
        # Step 10: Apply to values
        if use_einops:
            attn_output = einops.einsum(
                attn_pattern, v,
                "batch n_heads posn_q posn_k, batch posn_k n_heads d_head -> batch posn_q n_heads d_head"
            )
        else:
            v = v.transpose(1, 2)  # [batch, n_heads, posn, d_head]
            attn_output = torch.matmul(attn_pattern, v)  # [batch, n_heads, posn_q, d_head]
            attn_output = attn_output.transpose(1, 2)  # [batch, posn_q, n_heads, d_head]
        
        # Step 11: Output projection with LoRA
        output = compute_with_lora(
            attn_output, self.W_O,
            "batch posn n_heads d_head, n_heads d_head d_model -> batch posn d_model",
            "W_O"
        )
        
        # Return cache: use the original (non-broadcasted) K/V to save memory
        if return_attention_pattern:
            return output, (k_for_cache, v_for_cache), attn_pattern
        return output, (k_for_cache, v_for_cache)
    
    # Bind the method to the instance
    import types
    attn_module.forward = types.MethodType(lora_forward, attn_module)


def _patch_mlp_forward(mlp_module):
    """
    Patch MLP forward method to use LoRA.
    
    This dynamically replaces the forward method with a LoRA-aware version.
    Handles both standard GELU MLPs and SwiGLU MLPs (used in LLaMA).
    
    Args:
        mlp_module: MLP module to patch
    """
    # Check if SwiGLU (has W_gate and W_up instead of W_in)
    is_swiglu = hasattr(mlp_module, 'W_gate') and hasattr(mlp_module, 'W_up')
    
    def lora_forward(self, residual):
        """
        Forward pass with LoRA support.
        
        Applies LoRA to MLP layers (W_in/W_gate/W_up and W_out).
        Supports both GELU and SwiGLU activation functions.
        """
        def compute_with_lora(x, weight, pattern, bias, param_name):
            if hasattr(self, f'{param_name}_use_lora'):
                output = einsum_with_lora(
                    x, weight, pattern,
                    getattr(self, f'{param_name}_lora_A'),
                    getattr(self, f'{param_name}_lora_B'),
                    getattr(self, f'{param_name}_lora_scaling'),
                    getattr(self, f'{param_name}_lora_dropout'),
                )
                return output + bias
            else:
                # Use einops for consistency (works for both einops and non-einops MLPs)
                return einops.einsum(x, weight, pattern) + bias
        
        if is_swiglu:
            # SwiGLU MLP: gate and up branches
            gate = compute_with_lora(
                residual, self.W_gate,
                "batch posn d_model, d_model d_mlp -> batch posn d_mlp",
                self.b_gate, "W_gate"
            )
            gate = torch.nn.functional.silu(gate)  # Swish activation
            
            up = compute_with_lora(
                residual, self.W_up,
                "batch posn d_model, d_model d_mlp -> batch posn d_mlp",
                self.b_up, "W_up"
            )
            
            # Element-wise multiply
            hidden = gate * up
            
            # Output projection
            output = compute_with_lora(
                hidden, self.W_out,
                "batch posn d_mlp, d_mlp d_model -> batch posn d_model",
                self.b_out, "W_out"
            )
        else:
            # Standard GELU MLP
            # First layer with LoRA
            hidden = compute_with_lora(
                residual, self.W_in,
                "batch posn d_model, d_model d_mlp -> batch posn d_mlp",
                self.b_in, "W_in"
            )
            
            # GELU activation
            hidden = torch.nn.functional.gelu(hidden)
            
            # Second layer with LoRA
            output = compute_with_lora(
                hidden, self.W_out,
                "batch posn d_mlp, d_mlp d_model -> batch posn d_model",
                self.b_out, "W_out"
            )
        
        return output
    
    # Bind the method to the instance
    import types
    mlp_module.forward = types.MethodType(lora_forward, mlp_module)


def get_lora_parameters(model: nn.Module) -> list:
    """
    Get all LoRA parameters (A and B matrices) from a model.
    
    Args:
        model: Model with LoRA adapters
    
    Returns:
        List of LoRA parameters (only trainable ones)
    """
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora_A' in name or 'lora_B' in name:
            if param.requires_grad:
                lora_params.append(param)
    return lora_params


def count_lora_parameters(model: nn.Module) -> dict:
    """
    Count LoRA parameters vs total parameters.
    
    Returns:
        Dict with 'lora', 'total', 'trainable', 'frozen' counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora_params = sum(p.numel() for p in get_lora_parameters(model))
    frozen_params = total_params - trainable_params
    
    return {
        'lora': lora_params,
        'trainable': trainable_params,
        'frozen': frozen_params,
        'total': total_params,
    }

