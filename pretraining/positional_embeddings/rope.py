import torch
import torch.nn as nn
from jaxtyping import Float
from torch import Tensor


class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE) - LLaMA positional encoding"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_head = cfg.d_head
        self.theta = cfg.rope_theta

        # Pre-compute frequency matrix
        # Each dimension pair gets a different frequency
        # theta_i = theta^(-2i/d_head) for i in [0, d_head/2)
        freqs = 1.0 / (self.theta ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        self.register_buffer('freqs', freqs)

    def forward(
        self,
        q: Float[Tensor, "batch seq n_heads d_head"],
        k: Float[Tensor, "batch seq n_heads d_head"],
        positions: torch.Tensor  # [seq] - position indices
    ):
        """
        Apply rotary position embedding to queries and keys.

        Args:
            q: Query tensor [batch, seq, n_heads, d_head]
            k: Key tensor [batch, seq, n_heads, d_head]
            positions: Position indices [seq]

        Returns:
            Rotated q and k with same shapes
        """
        # q, k: [batch, seq, n_heads, d_head]
        batch, seq_len, n_heads, d_head = q.shape

        # Reshape to pairs: [batch, seq, n_heads, d_head/2, 2]
        # Each pair (x_i, x_i+1) will be rotated
        q_pairs = q.reshape(batch, seq_len, n_heads, d_head // 2, 2)
        k_pairs = k.reshape(batch, seq_len, n_heads, d_head // 2, 2)

        # Compute rotation angles for each position
        # angles: [seq, d_head/2]
        angles = positions.unsqueeze(-1) * self.freqs.unsqueeze(0)  # [seq, d_head/2]

        # Compute cos and sin
        cos = torch.cos(angles)  # [seq, d_head/2]
        sin = torch.sin(angles)  # [seq, d_head/2]

        # Expand for broadcasting: [1, seq, 1, d_head/2, 1]
        cos = cos.unsqueeze(0).unsqueeze(2).unsqueeze(-1)  # [1, seq, 1, d_head/2, 1]
        sin = sin.unsqueeze(0).unsqueeze(2).unsqueeze(-1)  # [1, seq, 1, d_head/2, 1]

        # Apply rotation to each pair
        # Rotation matrix: [cos(θ)  -sin(θ)]  [x]
        #                  [sin(θ)   cos(θ)]  [y]
        # cos and sin are [1, seq, 1, d_head/2, 1], need to squeeze last dim for broadcasting
        cos_squeezed = cos.squeeze(-1)  # [1, seq, 1, d_head/2]
        sin_squeezed = sin.squeeze(-1)  # [1, seq, 1, d_head/2]

        q_rotated = torch.stack([
            q_pairs[..., 0] * cos_squeezed - q_pairs[..., 1] * sin_squeezed,
            q_pairs[..., 0] * sin_squeezed + q_pairs[..., 1] * cos_squeezed
        ], dim=-1)  # [batch, seq, n_heads, d_head/2, 2]

        k_rotated = torch.stack([
            k_pairs[..., 0] * cos_squeezed - k_pairs[..., 1] * sin_squeezed,
            k_pairs[..., 0] * sin_squeezed + k_pairs[..., 1] * cos_squeezed
        ], dim=-1)  # [batch, seq, n_heads, d_head/2, 2]

        # Reshape back: [batch, seq, n_heads, d_head]
        q_rotated = q_rotated.reshape(batch, seq_len, n_heads, d_head)
        k_rotated = k_rotated.reshape(batch, seq_len, n_heads, d_head)

        return q_rotated, k_rotated

