"""Root Mean Square Normalization (RMSNorm) implementation.

This module implements RMSNorm (LLaMA style) which is a simpler normalization
than LayerNorm. RMSNorm only scales (divides by RMS), it doesn't center (no mean
subtraction) and has no bias term.

Design Decision: RMSNorm vs LayerNorm
- RMSNorm: Simpler (fewer operations), no bias term, works well in practice
- LayerNorm: More standard, centers data, has bias term
- LLaMA uses RMSNorm and shows it works as well as LayerNorm
- We provide both to show the difference

Mathematical Formula:
    RMSNorm(x) = γ * x / RMS(x)
    
Where:
    - RMS(x) = sqrt(mean(x²) + ε)
    - γ (w) = learnable scale parameter
    - ε = small constant for numerical stability (layer_norm_eps)
    - No bias term (unlike LayerNorm)

Why RMSNorm?
- Simpler: fewer operations (no mean subtraction, no bias)
- Faster: less computation per normalization
- Works well: LLaMA, PaLM, and other modern models use it
- No bias needed: scaling alone is sufficient
"""

import torch
import torch.nn as nn
import einops
from jaxtyping import Float
from torch import Tensor


class RMSNorm(nn.Module):
    """Root Mean Square Normalization (LLaMA style).
    
    Normalizes activations by scaling by RMS (Root Mean Square) instead of
    standard deviation. Unlike LayerNorm, RMSNorm:
    - Does NOT subtract mean (no centering)
    - Does NOT have bias term (only scale)
    - Only scales by RMS: x / RMS(x)
    
    This is simpler and faster than LayerNorm while working equally well.
    """
    
    def __init__(self, cfg, use_einops=True):
        """Initialize RMSNorm layer.
        
        Args:
            cfg: Model configuration
            use_einops: If True, use einops for tensor operations (more explicit),
                       else use PyTorch operations (more standard)
        """
        super().__init__()
        self.cfg = cfg
        self.use_einops = use_einops
        # w: [d_model] - learnable scale parameter (γ in formula)
        # No bias term (unlike LayerNorm)
        self.w = nn.Parameter(torch.ones(cfg.d_model))

    def _compute_rms(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn 1"]:
        """Compute RMS (Root Mean Square) over d_model dimension.
        
        Formula: RMS(x) = sqrt(mean(x²) + ε)
        Unlike LayerNorm, we don't subtract mean first.
        We compute mean of squared values, then take square root.
        
        Args:
            residual: Input tensor [batch, posn, d_model]
        
        Returns:
            RMS tensor [batch, posn, 1]
        """
        if self.use_einops:
            return torch.sqrt(
                einops.reduce(
                    residual ** 2,
                    'batch posn d_model -> batch posn 1',
                    'mean'
                ) + self.cfg.layer_norm_eps
            )
        else:
            return torch.sqrt(
                (residual ** 2).mean(dim=-1, keepdim=True) + self.cfg.layer_norm_eps
            )

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        """Forward pass through RMSNorm.
        
        Args:
            residual: Input tensor [batch, posn, d_model]
        
        Returns:
            Normalized tensor [batch, posn, d_model]
        """
        # Step 1: Compute RMS (Root Mean Square) over d_model dimension
        # Formula: RMS(x) = sqrt(mean(x²) + ε)
        rms = self._compute_rms(residual)

        # Step 2: Normalize: x / RMS(x)
        # This scales the data so RMS = 1
        # Unlike LayerNorm, we don't center (no mean subtraction)
        residual = residual / rms

        # Step 3: Apply learnable scale
        # Formula: output = γ * normalized
        # γ (w) allows the model to scale the normalized values
        # No bias term (unlike LayerNorm)
        return residual * self.w


# Backward compatibility aliases
RMSNormWithEinops = RMSNorm
RMSNormWithoutEinops = RMSNorm
