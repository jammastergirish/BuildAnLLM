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
        freqs = 1.0 / \
            (self.theta ** (torch.arange(0, self.d_head, 2).float() / self.d_head))
        self.register_buffer('freqs', freqs)

    def _compute_rotation_angles(
        self,
        positions: torch.Tensor
    ) -> tuple[Float[Tensor, "seq d_head/2"], Float[Tensor, "seq d_head/2"]]:
        """Compute rotation angles and their cos/sin values.

        Formula: angle = position * frequency
        Each position gets rotated by a different amount based on its position.

        Args:
            positions: Position indices [seq]

        Returns:
            Tuple of (cos, sin) where each is [seq, d_head/2]
        """
        # Compute rotation angles for each position
        # angles: [seq, d_head/2]
        angles = positions.unsqueeze(-1) * self.freqs.unsqueeze(0)

        # Compute cos and sin
        cos = torch.cos(angles)  # [seq, d_head/2]
        sin = torch.sin(angles)  # [seq, d_head/2]

        return cos, sin

    def _reshape_to_pairs(
        self,
        q_or_k: Float[Tensor, "batch seq n_heads d_head"]
    ) -> Float[Tensor, "batch seq n_heads d_head/2 2"]:
        """Reshape Q or K tensor to pairs for rotation.

        RoPE rotates pairs of dimensions: (x_i, x_i+1) together.
        This reshapes [batch, seq, n_heads, d_head] -> [batch, seq, n_heads, d_head/2, 2]

        Args:
            q_or_k: Query or Key tensor [batch, seq, n_heads, d_head]

        Returns:
            Reshaped tensor [batch, seq, n_heads, d_head/2, 2]
        """
        batch, seq_len, n_heads, d_head = q_or_k.shape
        return q_or_k.reshape(batch, seq_len, n_heads, d_head // 2, 2)

    def _apply_rotation(
        self,
        pairs: Float[Tensor, "batch seq n_heads d_head/2 2"],
        cos: Float[Tensor, "seq d_head/2"],
        sin: Float[Tensor, "seq d_head/2"]
    ) -> Float[Tensor, "batch seq n_heads d_head/2 2"]:
        """Apply rotation matrix to pairs.

        Rotation matrix: [cos(θ)  -sin(θ)]  [x]
                         [sin(θ)   cos(θ)]  [y]

        This rotates each pair (x, y) by angle θ.

        Args:
            pairs: Reshaped Q or K tensor [batch, seq, n_heads, d_head/2, 2]
            cos: Cosine values [seq, d_head/2]
            sin: Sine values [seq, d_head/2]

        Returns:
            Rotated pairs [batch, seq, n_heads, d_head/2, 2]
        """
        # Expand for broadcasting: [1, seq, 1, d_head/2]
        cos_expanded = cos.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, d_head/2]
        sin_expanded = sin.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, d_head/2]

        # Apply rotation matrix
        rotated = torch.stack([
            pairs[..., 0] * cos_expanded - pairs[..., 1] * sin_expanded,
            pairs[..., 0] * sin_expanded + pairs[..., 1] * cos_expanded
        ], dim=-1)  # [batch, seq, n_heads, d_head/2, 2]

        return rotated

    def _reshape_from_pairs(
        self,
        rotated_pairs: Float[Tensor, "batch seq n_heads d_head/2 2"],
        original_shape: tuple[int, int, int, int]
    ) -> Float[Tensor, "batch seq n_heads d_head"]:
        """Reshape rotated pairs back to original shape.

        Args:
            rotated_pairs: Rotated pairs [batch, seq, n_heads, d_head/2, 2]
            original_shape: Original shape (batch, seq, n_heads, d_head)

        Returns:
            Reshaped tensor [batch, seq, n_heads, d_head]
        """
        batch, seq_len, n_heads, d_head = original_shape
        return rotated_pairs.reshape(batch, seq_len, n_heads, d_head)

    def forward(
        self,
        q: Float[Tensor, "batch seq n_heads d_head"],
        k: Float[Tensor, "batch seq n_kv_heads d_head"],
        positions: torch.Tensor  # [seq] - position indices
    ):
        """
        Apply rotary position embedding to queries and keys.

        Args:
            q: Query tensor [batch, seq, n_heads, d_head]
            k: Key tensor [batch, seq, n_kv_heads, d_head] (supports GQA/MQA)
            positions: Position indices [seq]

        Returns:
            Rotated q and k with same shapes
        """
        batch, seq_len, n_heads_q, d_head = q.shape
        _, _, n_heads_k, _ = k.shape

        # Step 1: Reshape to pairs for rotation
        q_pairs = self._reshape_to_pairs(q)
        k_pairs = self._reshape_to_pairs(k)

        # Step 2: Compute rotation angles
        cos, sin = self._compute_rotation_angles(positions)

        # Step 3: Apply rotation to pairs
        q_rotated_pairs = self._apply_rotation(q_pairs, cos, sin)
        k_rotated_pairs = self._apply_rotation(k_pairs, cos, sin)

        # Step 4: Reshape back to original shape
        q_rotated = self._reshape_from_pairs(
            q_rotated_pairs, (batch, seq_len, n_heads_q, d_head))
        k_rotated = self._reshape_from_pairs(
            k_rotated_pairs, (batch, seq_len, n_heads_k, d_head))

        return q_rotated, k_rotated
