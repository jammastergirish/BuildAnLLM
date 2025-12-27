"""
LoRA wrappers for attention and MLP layers.

This module provides functions to apply LoRA (Low-Rank Adaptation) to transformer
components. Instead of modifying the original forward methods directly, it:
1. Adds LoRA matrices (A and B) as module attributes
2. Freezes the original weight parameters
3. The forward methods are patched in lora_utils.py to use these matrices

Key concepts:
- LoRA rank (r): Dimension of the low-rank matrices (typically 4-16)
- LoRA alpha (α): Scaling factor (typically = rank)
- LoRA adapters: Small trainable matrices that approximate weight updates
"""

import torch
import torch.nn as nn
import einops


def create_lora_matrices(weight_shape, rank: int, alpha: float, dropout: float = 0.0):
    """
    Create LoRA matrices A and B for a given weight shape.
    
    Args:
        weight_shape: Shape of the base weight
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout rate
    
    Returns:
        (lora_A, lora_B, scaling, dropout_layer)
    """
    scaling = alpha / rank
    
    if len(weight_shape) == 2:
        # Standard 2D: [out_dim, in_dim]
        out_dim, in_dim = weight_shape
        lora_A = nn.Parameter(torch.empty(rank, in_dim))
        lora_B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(lora_A, a=torch.sqrt(torch.tensor(5.0)))
    elif len(weight_shape) == 3:
        # 3D for attention: [n_heads, d_head, d_model]
        n_heads, d_head, d_model = weight_shape
        lora_A = nn.Parameter(torch.empty(n_heads, rank, d_model))
        lora_B = nn.Parameter(torch.zeros(n_heads, d_head, rank))
        nn.init.kaiming_uniform_(lora_A, a=torch.sqrt(torch.tensor(5.0)))
    else:
        raise ValueError(f"Unsupported weight shape: {weight_shape}")
    
    dropout_layer = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
    
    return lora_A, lora_B, scaling, dropout_layer


def apply_lora_to_attention(attn_module, rank: int = 8, alpha: float = 8.0, dropout: float = 0.0):
    """
    Apply LoRA to an attention module by adding LoRA matrices.
    
    Args:
        attn_module: Attention module (AttentionWithEinops or AttentionWithoutEinops)
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout rate
    """
    # Store original weights and freeze them
    for param_name in ['W_Q', 'W_K', 'W_V', 'W_O']:
        if hasattr(attn_module, param_name):
            weight = getattr(attn_module, param_name)
            weight.requires_grad = False
            
            # Create LoRA matrices
            lora_A, lora_B, scaling, dropout_layer = create_lora_matrices(
                weight.shape, rank, alpha, dropout
            )
            
            # Store as attributes
            setattr(attn_module, f'{param_name}_lora_A', lora_A)
            setattr(attn_module, f'{param_name}_lora_B', lora_B)
            setattr(attn_module, f'{param_name}_lora_scaling', scaling)
            setattr(attn_module, f'{param_name}_lora_dropout', dropout_layer)
            setattr(attn_module, f'{param_name}_use_lora', True)


def apply_lora_to_mlp(mlp_module, rank: int = 8, alpha: float = 8.0, dropout: float = 0.0):
    """
    Apply LoRA to an MLP module by adding LoRA matrices.
    
    Handles both standard MLP (W_in, W_out) and SwiGLU MLP (W_gate, W_up, W_out).
    
    Args:
        mlp_module: MLP module
        rank: LoRA rank
        alpha: LoRA alpha
        dropout: Dropout rate
    """
    # Check if SwiGLU (has W_gate and W_up)
    is_swiglu = hasattr(mlp_module, 'W_gate') and hasattr(mlp_module, 'W_up')
    
    if is_swiglu:
        # SwiGLU: apply to W_gate, W_up, W_out
        param_names = ['W_gate', 'W_up', 'W_out']
    else:
        # Standard: apply to W_in, W_out
        param_names = ['W_in', 'W_out']
    
    # Store original weights and freeze them
    for param_name in param_names:
        if hasattr(mlp_module, param_name):
            weight = getattr(mlp_module, param_name)
            weight.requires_grad = False
            
            # Create LoRA matrices
            lora_A, lora_B, scaling, dropout_layer = create_lora_matrices(
                weight.shape, rank, alpha, dropout
            )
            
            # Store as attributes
            setattr(mlp_module, f'{param_name}_lora_A', lora_A)
            setattr(mlp_module, f'{param_name}_lora_B', lora_B)
            setattr(mlp_module, f'{param_name}_lora_scaling', scaling)
            setattr(mlp_module, f'{param_name}_lora_dropout', dropout_layer)
            setattr(mlp_module, f'{param_name}_use_lora', True)


def einsum_with_lora(x, weight, pattern, lora_A, lora_B, scaling, dropout_layer):
    """
    Perform einsum with LoRA computation added.
    
    Args:
        x: Input tensor
        weight: Base weight (frozen)
        pattern: Einsum pattern
        lora_A: LoRA A matrix
        lora_B: LoRA B matrix
        scaling: LoRA scaling factor
        dropout_layer: Dropout layer
    
    Returns:
        Output with LoRA added
    """
    # Base computation
    base_output = einops.einsum(x, weight, pattern)
    
    # LoRA computation: compute B @ A first, then einsum
    if len(weight.shape) == 2:
        # Standard 2D: [out_dim, in_dim]
        # lora_B: [out_dim, rank], lora_A: [rank, in_dim]
        # [out_dim, rank] @ [rank, in_dim] = [out_dim, in_dim]
        lora_weight = torch.matmul(lora_B, lora_A)
    elif len(weight.shape) == 3:
        # 3D for attention: [n_heads, d_head, d_model]
        # lora_B: [n_heads, d_head, rank], lora_A: [n_heads, rank, d_model]
        # Batch matrix multiplication: [n_heads, d_head, rank] @ [n_heads, rank, d_model]
        lora_weight = torch.bmm(lora_B, lora_A)
    else:
        raise ValueError(f"Unsupported weight shape: {weight.shape}")
    
    # Apply LoRA using same einsum pattern
    lora_output = einops.einsum(x, lora_weight, pattern)
    lora_output = dropout_layer(lora_output)
    
    return base_output + scaling * lora_output

