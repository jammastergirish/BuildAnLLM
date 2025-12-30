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

    def _compute_distance_matrix(
        self,
        seq_len: int,
        device: torch.device
    ) -> Float[Tensor, "seq_len seq_len"]:
        """Compute distance matrix |i - j| between all position pairs.

        Args:
            seq_len: Sequence length
            device: Device to create matrix on

        Returns:
            Distance matrix [seq_len, seq_len] where distance[i, j] = |i - j|
        """
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)  # [seq_len]
        pos_i = positions.unsqueeze(1)  # [seq_len, 1]
        pos_j = positions.unsqueeze(0)  # [1, seq_len]
        return (pos_i - pos_j).abs()  # [seq_len, seq_len]

    def _apply_slopes(
        self,
        distance_matrix: Float[Tensor, "seq_len seq_len"]
    ) -> Float[Tensor, "n_heads seq_len seq_len"]:
        """Apply per-head slopes to distance matrix.

        Formula: bias[h, i, j] = -slope[h] * distance[i, j]
        Each head gets a different slope, creating different attention patterns.

        Args:
            distance_matrix: Distance matrix [seq_len, seq_len]

        Returns:
            Bias matrix [n_heads, seq_len, seq_len]
        """
        slopes_expanded = self.slopes.unsqueeze(
            -1).unsqueeze(-1)  # [n_heads, 1, 1]
        distance_expanded = distance_matrix.unsqueeze(
            0)  # [1, seq_len, seq_len]
        # [n_heads, seq_len, seq_len]
        return -slopes_expanded * distance_expanded

    def _apply_causal_mask(
        self,
        bias: Float[Tensor, "n_heads seq_len seq_len"],
        seq_len: int,
        device: torch.device
    ) -> Float[Tensor, "n_heads seq_len seq_len"]:
        """Apply causal mask to bias matrix.

        Sets bias to 0 for past/current positions (j <= i).
        Only future positions (j > i) get negative bias.

        Args:
            bias: Bias matrix [n_heads, seq_len, seq_len]
            seq_len: Sequence length
            device: Device for mask

        Returns:
            Masked bias matrix [n_heads, seq_len, seq_len]
        """
        # We want to apply the bias to all positions based on distance.
        # The causal mask will be applied later in the attention layer to mask out future positions.
        # So providing bias for future positions (which will be masked) doesn't hurt,
        # but crucially we MUST provide bias for past positions (which was being zeroed out).
        
        # Original incorrect logic: return bias * (1 - mask_expanded)
        # This zeroed out the valid positions (mask=1) and kept invalid ones (mask=0).
        
        # Correct logic: Just return the bias map.
        # bias[h, i, j] = -slope[h] * |i - j|
        return bias

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
        # Step 1: Compute distance matrix
        distance_matrix = self._compute_distance_matrix(seq_len, device)

        # Step 2: Apply per-head slopes
        bias = self._apply_slopes(distance_matrix)

        # Step 3: Apply causal mask
        bias = self._apply_causal_mask(bias, seq_len, device)

        return bias
