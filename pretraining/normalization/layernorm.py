"""Layer Normalization implementation.

This module implements LayerNorm (GPT/OLMo style) which normalizes activations
across the feature dimension. LayerNorm centers the data (subtracts mean) and
scales it (divides by standard deviation), then applies learnable scale and shift.

Design Decision: We provide both einops and PyTorch implementations to show
different approaches. Einops makes tensor operations explicit and readable,
while PyTorch operations are more standard.

Mathematical Formula:
    LayerNorm(x) = γ * (x - μ) / (σ + ε) + β
    
Where:
    - μ = mean(x) over d_model dimension
    - σ = std(x) over d_model dimension (unbiased=False, uses N not N-1)
    - γ (w) = learnable scale parameter
    - β (b) = learnable shift parameter
    - ε = small constant for numerical stability (layer_norm_eps)

Why LayerNorm?
- Stabilizes training by normalizing activations
- Allows higher learning rates
- Reduces internal covariate shift
- Essential for training deep networks
"""

import torch
import torch.nn as nn
import einops
from jaxtyping import Float
from torch import Tensor


class LayerNorm(nn.Module):
    """Layer Normalization (GPT/OLMo style).
    
    Normalizes activations across the feature dimension by:
    1. Computing mean and variance over d_model
    2. Centering: subtract mean
    3. Scaling: divide by standard deviation
    4. Applying learnable scale (γ) and shift (β)
    
    Design Decision: Pre-norm vs Post-norm
    - Pre-norm (what we use): x + f(LN(x)) - more stable for deep networks
    - Post-norm: LN(x + f(x)) - original transformer, less stable
    
    We use pre-norm because it allows training deeper networks without gradient issues.
    """
    
    def __init__(self, cfg, use_einops=True):
        """Initialize LayerNorm layer.
        
        Args:
            cfg: Model configuration
            use_einops: If True, use einops for tensor operations (more explicit),
                       else use PyTorch operations (more standard)
        """
        super().__init__()
        self.cfg = cfg
        self.use_einops = use_einops
        # w: [d_model] - learnable scale parameter (γ in formula)
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        # b: [d_model] - learnable bias parameter (β in formula)
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def _compute_mean(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn 1"]:
        """Compute mean over d_model dimension.
        
        Formula: μ = mean(x) over d_model
        
        Args:
            residual: Input tensor [batch, posn, d_model]
        
        Returns:
            Mean tensor [batch, posn, 1]
        """
        if self.use_einops:
            return einops.reduce(residual, 'batch posn d_model -> batch posn 1', 'mean')
        else:
            return residual.mean(dim=-1, keepdim=True)

    def _compute_variance(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn 1"]:
        """Compute variance over d_model dimension.
        
        Formula: σ² = var(x) over d_model
        Important: LayerNorm uses unbiased=False (N denominator, not N-1)
        This matches PyTorch's nn.LayerNorm behavior
        
        Args:
            residual: Input tensor [batch, posn, d_model]
        
        Returns:
            Variance tensor [batch, posn, 1]
        """
        if self.use_einops:
            def layernorm_variance(x, axis):
                """Variance for LayerNorm (uses N not N-1 denominator)"""
                return x.var(axis=axis, unbiased=False)
            return einops.reduce(residual, 'batch posn d_model -> batch posn 1', layernorm_variance)
        else:
            return residual.var(dim=-1, keepdim=True, unbiased=False)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        """Forward pass through LayerNorm.
        
        Args:
            residual: Input tensor [batch, posn, d_model]
        
        Returns:
            Normalized tensor [batch, posn, d_model]
        """
        # Step 1: Compute mean over d_model dimension
        # Formula: μ = mean(x) over d_model
        residual_mean = self._compute_mean(residual)

        # Step 2: Compute variance over d_model dimension
        # Formula: σ² = var(x) over d_model
        residual_variance = self._compute_variance(residual)

        # Step 3: Compute standard deviation
        # Formula: σ = sqrt(σ² + ε)
        # ε (epsilon) prevents division by zero and provides numerical stability
        residual_std = torch.sqrt(residual_variance + self.cfg.layer_norm_eps)

        # Step 4: Normalize: (x - μ) / σ
        # This centers the data (mean=0) and scales it (std=1)
        residual = (residual - residual_mean) / residual_std

        # Step 5: Apply learnable scale and shift
        # Formula: output = γ * normalized + β
        # γ (w) allows the model to scale the normalized values
        # β (b) allows the model to shift the normalized values
        # This gives the model flexibility: if normalization isn't needed,
        # it can learn γ≈std and β≈mean to undo normalization
        return residual * self.w + self.b


class LayerNormWithTorch(nn.Module):
    """LayerNorm using PyTorch's built-in implementation.
    
    This is the optimized, production-ready version. We include it to show
    that our manual implementation matches PyTorch's behavior.
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # PyTorch's built-in LayerNorm (optimized, handles edge cases)
        self.ln = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # residual: [batch, posn, d_model]
        # output: [batch, posn, d_model]
        return self.ln(residual)


# Backward compatibility aliases
LayerNormWithEinops = LayerNorm
LayerNormWithoutEinops = LayerNorm


def create_norm_layer(cfg, use_einops=True):
    """Factory function to create appropriate normalization layer.
    
    Args:
        cfg: Model configuration
        use_einops: If True, use einops implementations, else PyTorch
    
    Returns:
        Normalization layer (LayerNorm or RMSNorm based on cfg.normalization)
    """
    from config import Normalization

    if cfg.normalization == Normalization.RMSNORM:
        if use_einops:
            from pretraining.normalization.rmsnorm import RMSNorm
            return RMSNorm(cfg, use_einops=True)
        from pretraining.normalization.rmsnorm import RMSNorm
        return RMSNorm(cfg, use_einops=False)
    # LAYERNORM
    if use_einops:
        return LayerNorm(cfg, use_einops=True)
    return LayerNorm(cfg, use_einops=False)
