import torch
import torch.nn as nn
import einops
from jaxtyping import Float, Tensor


class MLPWithEinops(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
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

        # GELU activation
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
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
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

        # GELU activation
        hidden = torch.nn.functional.gelu(hidden)

        # Second linear layer: d_mlp -> d_model
        # hidden: [batch, posn, d_mlp]
        # W_out: [d_mlp, d_model]
        # output: [batch, posn, d_model]
        output = torch.matmul(hidden, self.W_out) + self.b_out

        return output
