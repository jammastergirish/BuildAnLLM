import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class ALiBi(nn.Module):
    """Attention with Linear Biases (ALiBi) - OLMo positional encoding

    ALiBi adds a linear bias to attention scores based on distance:
    - Closer positions get less negative bias
    - Farther positions get more negative bias
    - No learned parameters, computed on-the-fly
    - Each attention head gets a different slope
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads

        # Pre-compute slopes for each head
        # Slopes decrease geometrically: 2^(-8/n_heads * i) for head i
        # Each head gets a different slope
        # Head 0 gets the largest slope, head n_heads-1 gets the smallest
        slopes = torch.pow(2.0, -torch.arange(1, self.n_heads + 1,
                           dtype=torch.float32) * (8.0 / self.n_heads))
        self.register_buffer('slopes', slopes)  # [n_heads]

    def get_bias(self, seq_len: int, device: torch.device) -> Float[Tensor, "n_heads seq_len seq_len"]:
        """
        Compute ALiBi bias matrix.

        Args:
            seq_len: Sequence length
            device: Device to create bias on

        Returns:
            Bias matrix [n_heads, seq_len, seq_len]
            bias[h, i, j] = -slope[h] * |i - j| for j > i (future positions)
            bias[h, i, j] = 0 for j <= i (past/current positions)
        """
        # Create position indices
        positions = torch.arange(seq_len, device=device)  # [seq_len]

        # Compute distance matrix: |i - j|
        # positions: [seq_len] -> [seq_len, 1] and [1, seq_len]
        pos_i = positions.unsqueeze(1)  # [seq_len, 1]
        pos_j = positions.unsqueeze(0)  # [1, seq_len]
        # [seq_len, seq_len] - distance between positions
        distance = (pos_i - pos_j).abs()

        # Apply slopes: -slope[h] * distance
        # slopes: [n_heads] -> [n_heads, 1, 1]
        # distance: [seq_len, seq_len] -> [1, seq_len, seq_len]
        # [n_heads, 1, 1]
        slopes_expanded = self.slopes.unsqueeze(-1).unsqueeze(-1)
        distance_expanded = distance.unsqueeze(0)  # [1, seq_len, seq_len]

        # bias: [n_heads, seq_len, seq_len]
        bias = -slopes_expanded * distance_expanded

        # Mask future positions (causal): set bias to 0 for j <= i
        # Lower triangular mask: 1 for j <= i, 0 for j > i
        # [seq_len, seq_len]
        mask = torch.tril(torch.ones((seq_len, seq_len), device=device))
        mask_expanded = mask.unsqueeze(0)  # [1, seq_len, seq_len]

        # Only apply bias to future positions (where mask == 0)
        # For past/current positions (mask == 1), bias is 0
        bias = bias * (1 - mask_expanded)

        return bias  # [n_heads, seq_len, seq_len]
