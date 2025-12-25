import torch
import torch.nn as nn
import einops
from jaxtyping import Float
from torch import Tensor


class RMSNormWithEinops(nn.Module):
    """Root Mean Square Normalization (LLaMA style) with Einops"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # w: [d_model] - learnable scale parameter (no bias in RMSNorm)
        self.w = nn.Parameter(torch.ones(cfg.d_model))

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # residual: [batch, posn, d_model]

        # Compute RMS (Root Mean Square) over d_model dimension
        # rms: [batch, posn, 1]
        rms = torch.sqrt(
            einops.reduce(
                residual ** 2,
                'batch posn d_model -> batch posn 1',
                'mean'
            ) + self.cfg.layer_norm_eps
        )

        # Normalize: x / rms
        # residual: [batch, posn, d_model]
        residual = residual / rms

        # Apply learnable scale
        # w: [d_model]
        # output: [batch, posn, d_model]
        return residual * self.w


class RMSNormWithoutEinops(nn.Module):
    """Root Mean Square Normalization (LLaMA style) without Einops"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # w: [d_model] - learnable scale parameter (no bias in RMSNorm)
        self.w = nn.Parameter(torch.ones(cfg.d_model))

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # residual: [batch, posn, d_model]

        # Compute RMS (Root Mean Square) over d_model dimension
        # rms: [batch, posn, 1]
        rms = torch.sqrt(
            (residual ** 2).mean(dim=-1, keepdim=True) + self.cfg.layer_norm_eps
        )

        # Normalize: x / rms
        # residual: [batch, posn, d_model]
        residual = residual / rms

        # Apply learnable scale
        # w: [d_model]
        # output: [batch, posn, d_model]
        return residual * self.w

