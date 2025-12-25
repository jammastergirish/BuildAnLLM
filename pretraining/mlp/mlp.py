import torch
import torch.nn as nn
import einops
from jaxtyping import Float
from torch import Tensor


class MLPWithEinops(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # W_in: [d_model, d_mlp] - input projection
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        # b_in: [d_mlp] - input bias
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        # W_out: [d_mlp, d_model] - output projection
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        # b_out: [d_model] - output bias
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # First linear layer: d_model -> d_mlp
        # residual: [batch, posn, d_model]
        # W_in: [d_model, d_mlp]
        # hidden: [batch, posn, d_mlp]
        hidden = einops.einsum(
            residual, self.W_in,
            "batch posn d_model, d_model d_mlp -> batch posn d_mlp"
        ) + self.b_in

        # GELU activation (element-wise)
        # hidden: [batch, posn, d_mlp]
        hidden = torch.nn.functional.gelu(hidden)

        # Second linear layer: d_mlp -> d_model
        # hidden: [batch, posn, d_mlp]
        # W_out: [d_mlp, d_model]
        # output: [batch, posn, d_model]
        output = einops.einsum(
            hidden, self.W_out,
            "batch posn d_mlp, d_mlp d_model -> batch posn d_model"
        ) + self.b_out

        return output


class MLPWithoutEinops(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # W_in: [d_model, d_mlp] - input projection
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        # b_in: [d_mlp] - input bias
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        # W_out: [d_mlp, d_model] - output projection
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        # b_out: [d_model] - output bias
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # First linear layer: d_model -> d_mlp
        # residual: [batch, posn, d_model]
        # W_in: [d_model, d_mlp]
        # hidden: [batch, posn, d_mlp]
        hidden = torch.matmul(residual, self.W_in) + self.b_in

        # GELU activation (element-wise)
        # hidden: [batch, posn, d_mlp]
        hidden = torch.nn.functional.gelu(hidden)

        # Second linear layer: d_mlp -> d_model
        # hidden: [batch, posn, d_mlp]
        # W_out: [d_mlp, d_model]
        # output: [batch, posn, d_model]
        output = torch.matmul(hidden, self.W_out) + self.b_out

        return output


class MLPSwiGLUWithEinops(nn.Module):
    """MLP with SwiGLU activation (LLaMA style) using Einops"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # SwiGLU needs 3 weight matrices
        # W_gate: [d_model, d_mlp] - gate branch
        self.W_gate = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        # W_up: [d_model, d_mlp] - up branch
        self.W_up = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        # W_out: [d_mlp, d_model] - output projection
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        self.b_gate = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.b_up = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

        nn.init.normal_(self.W_gate, std=self.cfg.init_range)
        nn.init.normal_(self.W_up, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # residual: [batch, posn, d_model]

        # Gate branch: Swish activation (SiLU)
        # gate: [batch, posn, d_mlp]
        gate = torch.nn.functional.silu(
            einops.einsum(
                residual, self.W_gate,
                "batch posn d_model, d_model d_mlp -> batch posn d_mlp"
            ) + self.b_gate
        )  # SiLU = Swish

        # Up branch: linear
        # up: [batch, posn, d_mlp]
        up = einops.einsum(
            residual, self.W_up,
            "batch posn d_model, d_model d_mlp -> batch posn d_mlp"
        ) + self.b_up

        # Element-wise multiply (gating)
        # hidden: [batch, posn, d_mlp]
        hidden = gate * up

        # Output projection
        # output: [batch, posn, d_model]
        output = einops.einsum(
            hidden, self.W_out,
            "batch posn d_mlp, d_mlp d_model -> batch posn d_model"
        ) + self.b_out

        return output


class MLPSwiGLUWithoutEinops(nn.Module):
    """MLP with SwiGLU activation (LLaMA style) without Einops"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_gate = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        self.W_up = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        self.b_gate = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.b_up = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

        nn.init.normal_(self.W_gate, std=self.cfg.init_range)
        nn.init.normal_(self.W_up, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # residual: [batch, posn, d_model]

        # Gate branch: Swish activation (SiLU)
        gate = torch.nn.functional.silu(
            torch.matmul(residual, self.W_gate) + self.b_gate
        )  # [batch, posn, d_mlp]

        # Up branch: linear
        up = torch.matmul(residual, self.W_up) + \
            self.b_up  # [batch, posn, d_mlp]

        # Element-wise multiply (gating)
        hidden = gate * up  # [batch, posn, d_mlp]

        # Output projection
        output = torch.matmul(hidden, self.W_out) + \
            self.b_out  # [batch, posn, d_model]

        return output


def create_mlp_layer(cfg, use_einops=True):
    """Factory function to create appropriate MLP layer based on activation config"""
    from config import Activation

    if cfg.activation == Activation.SWIGLU:
        if use_einops:
            return MLPSwiGLUWithEinops(cfg)
        else:
            return MLPSwiGLUWithoutEinops(cfg)
    else:  # GELU
        if use_einops:
            return MLPWithEinops(cfg)
        else:
            return MLPWithoutEinops(cfg)
