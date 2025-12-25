import torch
import torch.nn as nn
import einops
from jaxtyping import Float, Int
from torch import Tensor


class PosEmbedWithEinops(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # W_pos: [n_ctx, d_model] - positional embedding matrix
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        # tokens: [batch, position]
        batch, seq_len = tokens.shape
        # W_pos[:seq_len]: [seq_len, d_model] - get embeddings for sequence length
        # einops.repeat: [seq_len, d_model] -> [batch, seq_len, d_model]
        return einops.repeat(
            self.W_pos[:seq_len], "seq d_model -> batch seq d_model", batch=batch
        )


class PosEmbedWithoutEinops(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # W_pos: [n_ctx, d_model] - positional embedding matrix
        self.W_pos = nn.Parameter(torch.empty((cfg.n_ctx, cfg.d_model)))
        nn.init.normal_(self.W_pos, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        # tokens: [batch, position]
        batch_size = tokens.shape[0]
        sequence_length = tokens.shape[1]

        # W_pos[:sequence_length]: [sequence_length, d_model]
        position_embeddings_we_need = self.W_pos[:sequence_length]

        # Manual repeat
        # Add a new dimension at the beginning
        # position_embeddings_with_batch_dim: [1, sequence_length, d_model]
        position_embeddings_with_batch_dim = position_embeddings_we_need.unsqueeze(
            0)

        # Repeat along the batch dimension
        # position_embeddings_repeated: [batch_size, sequence_length, d_model]
        position_embeddings_repeated = position_embeddings_with_batch_dim.expand(
            batch_size, -1, -1
        )
        return position_embeddings_repeated
