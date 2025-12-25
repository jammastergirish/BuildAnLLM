import torch
import torch.nn as nn
import einops
from jaxtyping import Float
from torch import Tensor


class LayerNormWithEinops(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # w: [d_model] - learnable scale parameter
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        # b: [d_model] - learnable bias parameter
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # residual: [batch, posn, d_model]

        # Compute mean over d_model dimension
        # residual_mean: [batch, posn, 1]
        residual_mean = einops.reduce(
            residual, 'batch posn d_model -> batch posn 1', 'mean')

        def layernorm_variance(x, axis):
            """Variance for LayerNorm (uses N not N-1 denominator)"""
            return x.var(axis=axis, unbiased=False)

        # Compute variance over d_model dimension
        # residual_variance: [batch, posn, 1]
        residual_variance = einops.reduce(
            residual, 'batch posn d_model -> batch posn 1', layernorm_variance)

        # Compute standard deviation (with epsilon for numerical stability)
        # residual_std: [batch, posn, 1]
        residual_std = torch.sqrt(residual_variance + self.cfg.layer_norm_eps)

        # Normalize: (x - mean) / std
        # residual: [batch, posn, d_model]
        residual = (residual - residual_mean) / residual_std

        # Apply learnable scale and shift
        # w: [d_model], b: [d_model]
        # output: [batch, posn, d_model]
        return residual * self.w + self.b


class LayerNormWithoutEinops(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # w: [d_model] - learnable scale parameter
        self.w = nn.Parameter(torch.ones(cfg.d_model))
        # b: [d_model] - learnable bias parameter
        self.b = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # residual: [batch, posn, d_model]

        # Compute mean over d_model dimension (last dim)
        # residual_mean: [batch, posn, 1]
        residual_mean = residual.mean(dim=-1, keepdim=True)

        # Compute variance over d_model dimension (unbiased=False for LayerNorm)
        # residual_variance: [batch, posn, 1]
        residual_variance = residual.var(dim=-1, keepdim=True, unbiased=False)

        # Compute standard deviation (with epsilon for numerical stability)
        # residual_std: [batch, posn, 1]
        residual_std = torch.sqrt(residual_variance + self.cfg.layer_norm_eps)

        # Normalize: (x - mean) / std
        # residual: [batch, posn, d_model]
        residual = (residual - residual_mean) / residual_std

        # Apply learnable scale and shift
        # w: [d_model], b: [d_model]
        # output: [batch, posn, d_model]
        return residual * self.w + self.b


class LayerNormWithTorch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # PyTorch's built-in LayerNorm
        self.ln = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # residual: [batch, posn, d_model]
        # output: [batch, posn, d_model]
        return self.ln(residual)


def create_norm_layer(cfg, use_einops=True):
    """Factory function to create appropriate normalization layer based on normalization config"""
    from config import Normalization

    if cfg.normalization == Normalization.RMSNORM:
        if use_einops:
            from pretraining.normalization.rmsnorm import RMSNormWithEinops
            return RMSNormWithEinops(cfg)
        from pretraining.normalization.rmsnorm import RMSNormWithoutEinops
        return RMSNormWithoutEinops(cfg)
    # LAYERNORM
    if use_einops:
        return LayerNormWithEinops(cfg)
    return LayerNormWithoutEinops(cfg)
