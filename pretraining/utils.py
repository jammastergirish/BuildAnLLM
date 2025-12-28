"""Shared utility functions for model output handling and auxiliary losses.

This module provides utilities for extracting logits and auxiliary losses from
model outputs, which can have different formats depending on whether:
- MoE (Mixture of Experts) is enabled (returns aux_loss)
- KV caching is enabled (returns cache)
- Both or neither are enabled

These utilities are used across training and inference code to handle the
various return formats consistently.
"""

from typing import Optional, Tuple, Union
from torch import Tensor
from jaxtyping import Float


def extract_model_output_and_aux_loss(
    model_output: Union[
        Float[Tensor, "batch position d_vocab"],
        Tuple[Float[Tensor, "batch position d_vocab"], Optional[Float[Tensor, ""]]],
        Tuple[Float[Tensor, "batch position d_vocab"], list],
        Tuple[Float[Tensor, "batch position d_vocab"], list, Optional[Float[Tensor, ""]]],
    ]
) -> Tuple[Float[Tensor, "batch position d_vocab"], Optional[Float[Tensor, ""]]]:
    """Extract logits and auxiliary loss from model output.
    
    Handles different return formats:
    - Standard: logits only
    - MoE: (logits, aux_loss)
    - With cache: (logits, cache) or (logits, cache, aux_loss)
    
    Args:
        model_output: Model forward pass output, which can be:
            - logits: [batch, position, d_vocab]
            - (logits, aux_loss): MoE without cache
            - (logits, cache): Standard with cache
            - (logits, cache, aux_loss): MoE with cache
    
    Returns:
        Tuple of (logits, aux_loss) where aux_loss is None if not present.
    """
    if isinstance(model_output, tuple):
        # Handle different return formats:
        # - (logits, cache, aux_loss) - MoE with cache
        # - (logits, cache) - standard with cache
        # - (logits, aux_loss) - MoE without cache
        if len(model_output) == 3:
            logits, _, aux_loss = model_output
        elif len(model_output) == 2:
            # Could be (logits, cache) or (logits, aux_loss)
            # Check if second element is a list (cache) or scalar/tensor (aux_loss)
            if isinstance(model_output[1], list):
                logits, _ = model_output
                aux_loss = None
            else:
                logits, aux_loss = model_output
        else:
            logits = model_output[0]
            aux_loss = None
    else:
        logits = model_output
        aux_loss = None
    return logits, aux_loss


def extract_mlp_output_and_aux_loss(
    mlp_output: Union[
        Float[Tensor, "batch posn d_model"],
        Tuple[Float[Tensor, "batch posn d_model"], Optional[Float[Tensor, ""]]]
    ]
) -> Tuple[Float[Tensor, "batch posn d_model"], Optional[Float[Tensor, ""]]]:
    """Extract output and auxiliary loss from MLP.
    
    Handles both standard MLPs (returns tensor) and MoE MLPs (returns tuple).
    
    Args:
        mlp_output: MLP forward pass output, which can be:
            - output: [batch, posn, d_model] for standard MLP
            - (output, aux_loss): [batch, posn, d_model], scalar for MoE MLP
    
    Returns:
        Tuple of (output, aux_loss) where aux_loss is None for standard MLPs.
    """
    aux_loss = None
    if isinstance(mlp_output, tuple):
        mlp_output, aux_loss = mlp_output
    return mlp_output, aux_loss


def add_aux_loss_to_main_loss(
    loss: Float[Tensor, ""],
    aux_loss: Optional[Float[Tensor, ""]],
    model
) -> Float[Tensor, ""]:
    """Add auxiliary loss to main loss if MoE is enabled.
    
    MoE models use an auxiliary load balancing loss to encourage uniform
    expert usage. This function adds it to the main loss with the configured
    weight from the model config.
    
    Args:
        loss: Main training loss (cross-entropy)
        aux_loss: Auxiliary loss from MoE (load balancing), None if not MoE
        model: Model instance (needed to access cfg.load_balancing_loss_weight)
    
    Returns:
        Combined loss (main + weighted auxiliary) if aux_loss exists, else main loss.
    """
    if aux_loss is not None:
        cfg = model.cfg if hasattr(model, "cfg") else None
        if cfg and hasattr(cfg, "load_balancing_loss_weight"):
            loss = loss + aux_loss * cfg.load_balancing_loss_weight
    return loss

