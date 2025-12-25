import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor


class EmbedWithoutTorch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # W_E: [d_vocab, d_model] - embedding matrix
        self.W_E = nn.Parameter(torch.empty((cfg.d_vocab, cfg.d_model)))
        nn.init.normal_(self.W_E, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        # tokens: [batch, position] - token IDs
        # W_E: [d_vocab, d_model]
        # W_E[tokens]: [batch, position, d_model] - indexed embeddings
        return self.W_E[tokens]


class EmbedWithTorch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # embedding.weight: [d_vocab, d_model]
        self.embedding = nn.Embedding(cfg.d_vocab, cfg.d_model)
        nn.init.normal_(self.embedding.weight, std=self.cfg.init_range)

    def forward(
        self, tokens: Int[Tensor, "batch position"]
    ) -> Float[Tensor, "batch position d_model"]:
        # tokens: [batch, position] - token IDs
        # embedding(tokens): [batch, position, d_model]
        return self.embedding(tokens)


class UnembedWithoutTorch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # W_U: [d_model, d_vocab] - unembedding matrix
        self.W_U = nn.Parameter(torch.empty((cfg.d_model, cfg.d_vocab)))
        nn.init.normal_(self.W_U, std=self.cfg.init_range)

    def forward(
        self, residual: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        # residual: [batch, position, d_model]
        # W_U: [d_model, d_vocab]
        # logits: [batch, position, d_vocab]
        return torch.matmul(residual, self.W_U)


class UnembedWithTorch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # linear.weight: [d_vocab, d_model] (transposed internally)
        # linear: [d_model] -> [d_vocab]
        self.linear = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)
        nn.init.normal_(self.linear.weight, std=self.cfg.init_range)

    def forward(
        self, residual: Float[Tensor, "batch position d_model"]
    ) -> Float[Tensor, "batch position d_vocab"]:
        # residual: [batch, position, d_model]
        # linear(residual): [batch, position, d_vocab]
        return self.linear(residual)
