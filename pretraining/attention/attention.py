import torch
import torch.nn as nn
import einops
from jaxtyping import Float
from torch import Tensor


class AttentionWithEinops(nn.Module):
    def __init__(self, cfg, rope=None, alibi=None):
        super().__init__()
        self.cfg = cfg
        self.rope = rope  # RoPE module (None for GPT, RoPE instance for LLaMA)
        # ALiBi module (None for GPT/LLaMA, ALiBi instance for OLMo)
        self.alibi = alibi
        self.W_Q = nn.Parameter(torch.empty(
            (cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.W_K = nn.Parameter(torch.empty(
            (cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.W_V = nn.Parameter(torch.empty(
            (cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.W_O = nn.Parameter(torch.empty(
            (cfg.n_heads, cfg.d_head, cfg.d_model)))

        for param in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.normal_(param, std=self.cfg.init_range)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # residual: [batch, posn, d_model]
        seq_len = residual.shape[1]

        # Compute Q, K, V for all heads
        # residual: [batch, posn, d_model]
        # W_Q: [n_heads, d_head, d_model]
        # q: [batch, posn, n_heads, d_head]
        q = einops.einsum(
            residual, self.W_Q,
            "batch posn d_model, n_heads d_head d_model -> batch posn n_heads d_head"
        )
        # k: [batch, posn, n_heads, d_head]
        k = einops.einsum(
            residual, self.W_K,
            "batch posn d_model, n_heads d_head d_model -> batch posn n_heads d_head"
        )
        # v: [batch, posn, n_heads, d_head]
        v = einops.einsum(
            residual, self.W_V,
            "batch posn d_model, n_heads d_head d_model -> batch posn n_heads d_head"
        )

        # Apply RoPE if provided (LLaMA)
        if self.rope is not None:
            positions = torch.arange(seq_len, device=residual.device)
            q, k = self.rope(q, k, positions)

        # Scaled dot-product attention
        # q: [batch, posn_q, n_heads, d_head]
        # k: [batch, posn_k, n_heads, d_head]
        # attn_scores: [batch, n_heads, posn_q, posn_k]
        attn_scores = einops.einsum(
            q, k,
            "batch posn_q n_heads d_head, batch posn_k n_heads d_head -> batch n_heads posn_q posn_k"
        ) / (self.cfg.d_head ** 0.5)

        # Apply ALiBi bias if provided (OLMo)
        if self.alibi is not None:
            alibi_bias = self.alibi.get_bias(
                seq_len, residual.device)  # [n_heads, seq_len, seq_len]
            # Add bias to attention scores
            # [batch, n_heads, posn_q, posn_k]
            attn_scores = attn_scores + alibi_bias.unsqueeze(0)

        # Causal mask: mask out future positions
        # mask: [seq_len, seq_len] - lower triangular matrix
        mask = torch.tril(torch.ones(
            (seq_len, seq_len), device=residual.device))
        # attn_scores: [batch, n_heads, posn_q, posn_k] - masked
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax over last dimension (keys)
        # attn_pattern: [batch, n_heads, posn_q, posn_k]
        attn_pattern = torch.softmax(attn_scores, dim=-1)

        # Apply attention to values
        # attn_pattern: [batch, n_heads, posn_q, posn_k]
        # v: [batch, posn_k, n_heads, d_head]
        # attn_output: [batch, posn_q, n_heads, d_head]
        attn_output = einops.einsum(
            attn_pattern, v,
            "batch n_heads posn_q posn_k, batch posn_k n_heads d_head -> batch posn_q n_heads d_head"
        )

        # Project back to d_model
        # attn_output: [batch, posn, n_heads, d_head]
        # W_O: [n_heads, d_head, d_model]
        # output: [batch, posn, d_model]
        output = einops.einsum(
            attn_output, self.W_O,
            "batch posn n_heads d_head, n_heads d_head d_model -> batch posn d_model"
        )

        return output


class AttentionWithoutEinops(nn.Module):
    def __init__(self, cfg, rope=None, alibi=None):
        super().__init__()
        self.cfg = cfg
        self.rope = rope  # RoPE module (None for GPT, RoPE instance for LLaMA)
        # ALiBi module (None for GPT/LLaMA, ALiBi instance for OLMo)
        self.alibi = alibi
        # W_Q, W_K, W_V: [n_heads, d_head, d_model] - per-head projection matrices
        self.W_Q = nn.Parameter(torch.empty(
            (cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.W_K = nn.Parameter(torch.empty(
            (cfg.n_heads, cfg.d_head, cfg.d_model)))
        self.W_V = nn.Parameter(torch.empty(
            (cfg.n_heads, cfg.d_head, cfg.d_model)))
        # W_O: [n_heads, d_head, d_model] - output projection
        self.W_O = nn.Parameter(torch.empty(
            (cfg.n_heads, cfg.d_head, cfg.d_model)))

        for param in [self.W_Q, self.W_K, self.W_V, self.W_O]:
            nn.init.normal_(param, std=self.cfg.init_range)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        # residual: [batch, posn, d_model]
        seq_len = residual.shape[1]

        # Compute Q, K, V for all heads
        # residual: [batch, seq_len, d_model]
        # W_Q: [n_heads, d_head, d_model]
        # q: [batch, seq_len, n_heads, d_head]
        q = torch.einsum("bpd,nhd->bpnh", residual, self.W_Q)
        # k: [batch, seq_len, n_heads, d_head]
        k = torch.einsum("bpd,nhd->bpnh", residual, self.W_K)
        # v: [batch, seq_len, n_heads, d_head]
        v = torch.einsum("bpd,nhd->bpnh", residual, self.W_V)

        # Apply RoPE if provided (LLaMA)
        if self.rope is not None:
            positions = torch.arange(seq_len, device=residual.device)
            q, k = self.rope(q, k, positions)

        # Scaled dot-product attention
        # Transpose to [batch, n_heads, seq_len, d_head] for matmul
        # q: [batch, seq_len, n_heads, d_head] -> [batch, n_heads, seq_len, d_head]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # attn_scores: [batch, n_heads, seq_len, seq_len]
        # q @ k^T: [batch, n_heads, seq_len, d_head] @ [batch, n_heads, d_head, seq_len]
        attn_scores = torch.matmul(
            q, k.transpose(-2, -1)) / (self.cfg.d_head ** 0.5)

        # Apply ALiBi bias if provided (OLMo)
        if self.alibi is not None:
            alibi_bias = self.alibi.get_bias(
                seq_len, residual.device)  # [n_heads, seq_len, seq_len]
            # Add bias to attention scores
            # [batch, n_heads, seq_len, seq_len]
            attn_scores = attn_scores + alibi_bias.unsqueeze(0)

        # Causal mask: mask out future positions
        # mask: [seq_len, seq_len] - lower triangular matrix
        mask = torch.tril(torch.ones(
            (seq_len, seq_len), device=residual.device))
        # attn_scores: [batch, n_heads, seq_len, seq_len] - masked
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        # Softmax over last dimension (keys)
        # attn_pattern: [batch, n_heads, seq_len, seq_len]
        attn_pattern = torch.softmax(attn_scores, dim=-1)

        # Apply attention to values
        # attn_pattern: [batch, n_heads, seq_len, seq_len]
        # v: [batch, n_heads, seq_len, d_head]
        # attn_output: [batch, n_heads, seq_len, d_head]
        attn_output = torch.matmul(attn_pattern, v)

        # Concatenate heads: [batch, n_heads, seq_len, d_head] -> [batch, seq_len, n_heads, d_head]
        attn_output = attn_output.transpose(1, 2)

        # Project back to d_model
        # attn_output: [batch, seq_len, n_heads, d_head]
        # W_O: [n_heads, d_head, d_model]
        # output: [batch, seq_len, d_model]
        output = torch.einsum("bpnh,nhd->bpd", attn_output, self.W_O)

        return output
