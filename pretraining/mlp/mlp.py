"""Multi-Layer Perceptron (MLP) / Feedforward Network implementations.

This module implements the feedforward networks used in transformer blocks.
It supports:
- GELU activation (GPT style): Standard 2-layer MLP with GELU
- SwiGLU activation (LLaMA/OLMo style): Gated MLP with Swish activation
- MoE (Mixture of Experts): Multiple expert MLPs with routing

Design Decision: GELU vs SwiGLU
- GELU: Standard activation, used in GPT-2, BERT
- SwiGLU: More expressive, allows model to gate information flow
- SwiGLU uses 3 weight matrices (gate, up, out) vs 2 for GELU
- Modern models (LLaMA, PaLM, OLMo) use SwiGLU for better performance

Mathematical Formulas:
    GELU MLP: output = W_out @ GELU(W_in @ x + b_in) + b_out
    SwiGLU MLP: output = W_out @ (Swish(W_gate @ x + b_gate) * (W_up @ x + b_up)) + b_out
    
Where:
    - GELU(x) = x * Φ(x) where Φ is CDF of standard normal
    - Swish(x) = x * sigmoid(x) (also called SiLU)
    - * denotes element-wise multiplication (gating)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from jaxtyping import Float
from torch import Tensor
from typing import Optional, Tuple
from config import RouterType, Activation


class MLP(nn.Module):
    """Standard MLP with GELU activation (GPT style).

    This is a 2-layer feedforward network that expands to d_mlp (typically 4x d_model)
    and then projects back to d_model. Each position is processed independently.

    Design Decision: Why expand to d_mlp?
    - More capacity: d_mlp = 4 * d_model gives model more parameters
    - Non-linearity: GELU provides non-linear transformation
    - Position-wise: Each position processed independently (no cross-position interaction)
    """

    def __init__(self, cfg, use_einops=True):
        """Initialize MLP layer.

        Args:
            cfg: Model configuration
            use_einops: If True, use einops for tensor operations (more explicit),
                       else use PyTorch operations (more standard)
        """
        super().__init__()
        self.cfg = cfg
        self.use_einops = use_einops
        # W_in: [d_model, d_mlp] - input projection (expands dimension)
        self.W_in = nn.Parameter(torch.empty((cfg.d_model, cfg.d_mlp)))
        # b_in: [d_mlp] - input bias
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))
        # W_out: [d_mlp, d_model] - output projection (contracts back)
        self.W_out = nn.Parameter(torch.empty((cfg.d_mlp, cfg.d_model)))
        # b_out: [d_model] - output bias
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))

        nn.init.normal_(self.W_in, std=self.cfg.init_range)
        nn.init.normal_(self.W_out, std=self.cfg.init_range)

    def _compute_hidden(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_mlp"]:
        """Compute hidden representation after first linear layer.

        Formula: hidden = W_in @ x + b_in

        Args:
            residual: Input tensor [batch, posn, d_model]

        Returns:
            Hidden tensor [batch, posn, d_mlp]
        """
        if self.use_einops:
            return einops.einsum(
                residual, self.W_in,
                "batch posn d_model, d_model d_mlp -> batch posn d_mlp"
            ) + self.b_in
        else:
            return torch.matmul(residual, self.W_in) + self.b_in

    def _project_output(self, hidden: Float[Tensor, "batch posn d_mlp"]) -> Float[Tensor, "batch posn d_model"]:
        """Project hidden representation back to d_model.

        Formula: output = W_out @ hidden + b_out

        Args:
            hidden: Hidden tensor [batch, posn, d_mlp]

        Returns:
            Output tensor [batch, posn, d_model]
        """
        if self.use_einops:
            return einops.einsum(
                hidden, self.W_out,
                "batch posn d_mlp, d_mlp d_model -> batch posn d_model"
            ) + self.b_out
        else:
            return torch.matmul(hidden, self.W_out) + self.b_out

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        """Forward pass through MLP.

        Args:
            residual: Input tensor [batch, posn, d_model]

        Returns:
            Output tensor [batch, posn, d_model]
        """
        # Step 1: First linear layer: d_model -> d_mlp
        # Formula: hidden = W_in @ x + b_in
        hidden = self._compute_hidden(residual)

        # Step 2: GELU activation (element-wise)
        # Formula: GELU(x) = x * Φ(x) where Φ is CDF of standard normal
        # GELU is smoother than ReLU and works better for transformers
        hidden = torch.nn.functional.gelu(hidden)

        # Step 3: Second linear layer: d_mlp -> d_model
        # Formula: output = W_out @ hidden + b_out
        return self._project_output(hidden)


class MLPSwiGLU(nn.Module):
    """MLP with SwiGLU activation (LLaMA/OLMo style).

    SwiGLU is a gated activation function that uses two branches:
    - Gate branch: Swish(W_gate @ x + b_gate) - controls information flow
    - Up branch: W_up @ x + b_up - provides values to gate

    The output is element-wise multiplication: gate * up

    Design Decision: Why SwiGLU?
    - More expressive: Gating allows model to control information flow
    - Better performance: LLaMA, PaLM, OLMo all use SwiGLU
    - Uses 3 weight matrices instead of 2 (more parameters, better capacity)
    """

    def __init__(self, cfg, use_einops=True):
        """Initialize SwiGLU MLP layer.

        Args:
            cfg: Model configuration
            use_einops: If True, use einops for tensor operations (more explicit),
                       else use PyTorch operations (more standard)
        """
        super().__init__()
        self.cfg = cfg
        self.use_einops = use_einops
        # SwiGLU needs 3 weight matrices (vs 2 for GELU)
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

    def _compute_gate(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_mlp"]:
        """Compute gate branch with Swish activation.

        Formula: gate = Swish(W_gate @ x + b_gate)
        Swish(x) = x * sigmoid(x) (also called SiLU)

        Args:
            residual: Input tensor [batch, posn, d_model]

        Returns:
            Gate tensor [batch, posn, d_mlp]
        """
        if self.use_einops:
            gate_pre_act = einops.einsum(
                residual, self.W_gate,
                "batch posn d_model, d_model d_mlp -> batch posn d_mlp"
            ) + self.b_gate
        else:
            gate_pre_act = torch.matmul(residual, self.W_gate) + self.b_gate
        return torch.nn.functional.silu(gate_pre_act)

    def _compute_up(self, residual: Float[Tensor, "batch posn d_model"]) -> Float[Tensor, "batch posn d_mlp"]:
        """Compute up branch (linear, no activation).

        Formula: up = W_up @ x + b_up

        Args:
            residual: Input tensor [batch, posn, d_model]

        Returns:
            Up tensor [batch, posn, d_mlp]
        """
        if self.use_einops:
            return einops.einsum(
                residual, self.W_up,
                "batch posn d_model, d_model d_mlp -> batch posn d_mlp"
            ) + self.b_up
        else:
            return torch.matmul(residual, self.W_up) + self.b_up

    def _project_output(self, hidden: Float[Tensor, "batch posn d_mlp"]) -> Float[Tensor, "batch posn d_model"]:
        """Project hidden representation back to d_model.

        Formula: output = W_out @ hidden + b_out

        Args:
            hidden: Hidden tensor [batch, posn, d_mlp]

        Returns:
            Output tensor [batch, posn, d_model]
        """
        if self.use_einops:
            return einops.einsum(
                hidden, self.W_out,
                "batch posn d_mlp, d_mlp d_model -> batch posn d_model"
            ) + self.b_out
        else:
            return torch.matmul(hidden, self.W_out) + self.b_out

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Float[Tensor, "batch posn d_model"]:
        """Forward pass through SwiGLU MLP.

        Args:
            residual: Input tensor [batch, posn, d_model]

        Returns:
            Output tensor [batch, posn, d_model]
        """
        # Step 1: Gate branch with Swish activation (SiLU)
        # Formula: gate = Swish(W_gate @ x + b_gate)
        gate = self._compute_gate(residual)

        # Step 2: Up branch (linear, no activation)
        # Formula: up = W_up @ x + b_up
        up = self._compute_up(residual)

        # Step 3: Element-wise multiply (gating)
        # Formula: hidden = gate * up
        # The gate controls how much information flows through
        hidden = gate * up

        # Step 4: Output projection
        # Formula: output = W_out @ hidden + b_out
        return self._project_output(hidden)


class MoEMLPBase(nn.Module):
    """Base class for Mixture of Experts MLP.

    MoE scales model capacity efficiently by using multiple expert MLPs and
    routing tokens to a subset of experts. For each token, a router selects
    the top-k experts to activate, allowing the model to have more parameters
    while keeping computation per token similar.

    Design Decision: Why MoE?
    - Scalability: Can have many experts (e.g., 64) while only activating a few per token
    - Efficiency: Computation scales with activated experts, not total experts
    - Specialization: Different experts can specialize in different patterns

    Mathematical Formula (Load Balancing Loss):
        aux_loss = num_experts * Σ_i (P_i * f_i)

    Where:
        - P_i = average routing probability for expert i
        - f_i = fraction of tokens routed to expert i
        - This encourages uniform expert usage (prevents expert collapse)
    """

    def __init__(self, cfg, expert_class):
        """Initialize MoE MLP layer.

        Args:
            cfg: Model configuration
            expert_class: Class to use for expert MLPs (MLP or MLPSwiGLU)
        """
        super().__init__()
        self.cfg = cfg
        self.num_experts = cfg.num_experts
        self.num_experts_per_tok = cfg.num_experts_per_tok
        self.use_shared_experts = cfg.use_shared_experts if cfg.use_moe else False
        self.num_shared_experts = cfg.num_shared_experts if self.use_shared_experts else 0
        self.router_type = cfg.router_type if cfg.use_moe else RouterType.TOP_K

        # Router network: [d_model] -> [num_experts]
        # Computes logits for each expert (which expert should process this token?)
        self.router = nn.Linear(cfg.d_model, cfg.num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=cfg.init_range)

        # Create expert MLPs (each is a standard MLP or SwiGLU MLP)
        self.experts = nn.ModuleList([
            expert_class(cfg) for _ in range(cfg.num_experts)
        ])

        # Shared experts (if enabled, DeepSeek-style)
        # Shared experts are always active (not routed)
        if self.use_shared_experts:
            self.shared_experts = nn.ModuleList([
                expert_class(cfg) for _ in range(self.num_shared_experts)
            ])
        else:
            self.shared_experts = None

    def _compute_load_balancing_loss(
        self, router_probs: Float[Tensor, "batch seq_len num_experts"],
        top_k_indices: Float[Tensor, "batch seq_len num_experts_per_tok"],
        batch_size: int, seq_len: int
    ) -> Float[Tensor, ""]:
        """Compute load balancing auxiliary loss.

        This loss encourages uniform expert usage to prevent expert collapse
        (where only a few experts are used).

        Formula: aux_loss = num_experts * Σ_i (P_i * f_i)
        Where P_i is average routing probability and f_i is fraction of tokens.

        Args:
            router_probs: Routing probabilities [batch, seq_len, num_experts]
            top_k_indices: Indices of top-k experts per token [batch, seq_len, num_experts_per_tok]
            batch_size: Batch size
            seq_len: Sequence length

        Returns:
            Scalar auxiliary loss
        """
        # Calculate fraction of tokens routed to each expert
        # f_i = (number of tokens routed to expert i) / (total tokens * num_experts_per_tok)
        expert_usage = torch.zeros(
            self.num_experts, device=router_probs.device)
        for k in range(self.num_experts_per_tok):
            for expert_idx in range(self.num_experts):
                mask = (top_k_indices[:, :, k] == expert_idx)
                expert_usage[expert_idx] += mask.sum().float()
        expert_usage = expert_usage / \
            (batch_size * seq_len * self.num_experts_per_tok)

        # Average routing probability: P_i = mean(router_probs[:, :, i])
        avg_router_probs = router_probs.mean(dim=[0, 1])  # [num_experts]

        # Load balancing loss: num_experts * sum(P_i * f_i)
        # This is minimized when P_i ≈ f_i (uniform usage)
        return self.num_experts * torch.sum(avg_router_probs * expert_usage)

    def _compute_routing(
        self,
        residual: Float[Tensor, "batch posn d_model"]
    ) -> tuple[Float[Tensor, "batch seq_len num_experts"], Float[Tensor, "batch seq_len num_experts_per_tok"], Float[Tensor, "batch seq_len num_experts_per_tok"]]:
        """Compute routing probabilities and select top-k experts.

        Args:
            residual: Input tensor [batch, posn, d_model]

        Returns:
            Tuple of (router_probs, top_k_probs, top_k_indices) where:
            - router_probs: [batch, seq_len, num_experts] - routing probabilities
            - top_k_probs: [batch, seq_len, num_experts_per_tok] - normalized top-k probabilities
            - top_k_indices: [batch, seq_len, num_experts_per_tok] - expert indices
        """
        # Step 1: Router computes logits for each expert
        router_logits = self.router(residual)  # [batch, seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Step 2: Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(
            router_probs, k=self.num_experts_per_tok, dim=-1
        )

        # Step 3: Normalize top-k probabilities
        # Renormalize so probabilities sum to 1 over selected experts
        top_k_probs = top_k_probs / \
            (top_k_probs.sum(dim=-1, keepdim=True) + 1e-8)

        return router_probs, top_k_probs, top_k_indices

    def _process_experts(
        self,
        residual: Float[Tensor, "batch posn d_model"],
        router_probs: Float[Tensor, "batch seq_len num_experts"],
        top_k_probs: Float[Tensor, "batch seq_len num_experts_per_tok"],
        top_k_indices: Float[Tensor, "batch seq_len num_experts_per_tok"]
    ) -> Float[Tensor, "batch posn d_model"]:
        """Process routed experts and compute weighted output.

        Only computes outputs for experts that are selected (sparse activation).

        Args:
            residual: Input tensor [batch, posn, d_model]
            router_probs: Routing probabilities [batch, seq_len, num_experts]
            top_k_probs: Normalized top-k probabilities [batch, seq_len, num_experts_per_tok]
            top_k_indices: Expert indices [batch, seq_len, num_experts_per_tok]

        Returns:
            Routed expert output [batch, posn, d_model]
        """
        output = torch.zeros_like(residual)

        for expert_idx in range(self.num_experts):
            # Find tokens that use this expert
            expert_mask = (top_k_indices == expert_idx).any(
                dim=-1)  # [batch, seq_len]

            if expert_mask.any():
                # Get expert output (only compute for tokens that use this expert)
                expert_output = self.experts[expert_idx](
                    residual)  # [batch, seq_len, d_model]

                # Get routing weights for this expert
                expert_weights = torch.zeros_like(
                    router_probs[:, :, expert_idx])
                for k in range(self.num_experts_per_tok):
                    mask = (top_k_indices[:, :, k] == expert_idx)
                    expert_weights[mask] = top_k_probs[:, :, k][mask]

                # Weighted contribution: output += weight * expert_output
                output += expert_weights.unsqueeze(-1) * expert_output

        return output

    def _process_shared_experts(
        self,
        residual: Float[Tensor, "batch posn d_model"]
    ) -> Optional[Float[Tensor, "batch posn d_model"]]:
        """Process shared experts if enabled (DeepSeek-style).

        Shared experts are always active (not routed). They handle general knowledge
        while routed experts specialize based on input.

        Args:
            residual: Input tensor [batch, posn, d_model]

        Returns:
            Shared expert output [batch, posn, d_model] or None if disabled
        """
        if not (self.use_shared_experts and self.shared_experts is not None):
            return None

        shared_output = torch.zeros_like(residual)
        for shared_expert in self.shared_experts:
            shared_output += shared_expert(residual)
        return shared_output / len(self.shared_experts)

    def forward(
        self, residual: Float[Tensor, "batch posn d_model"]
    ) -> Tuple[Float[Tensor, "batch posn d_model"], Optional[Float[Tensor, ""]]]:
        """Forward pass through MoE MLP.

        Args:
            residual: Input tensor [batch, posn, d_model]

        Returns:
            Tuple of (output, aux_loss) where:
            - output: [batch, posn, d_model] - MoE output
            - aux_loss: Scalar load balancing loss (None if not training)
        """
        batch_size, seq_len, d_model = residual.shape

        # Step 1: Compute routing probabilities and select top-k experts
        router_probs, top_k_probs, top_k_indices = self._compute_routing(
            residual)

        # Step 2: Process routed experts (sparse activation)
        output = self._process_experts(
            residual, router_probs, top_k_probs, top_k_indices)

        # Step 3: Add shared experts contribution (if enabled, DeepSeek-style)
        # Shared experts are always active (not routed)
        shared_output = self._process_shared_experts(residual)
        if shared_output is not None:
            output += shared_output

        # Step 4: Calculate load balancing loss (only during training)
        aux_loss = None
        if self.training:
            aux_loss = self._compute_load_balancing_loss(
                router_probs, top_k_indices, batch_size, seq_len
            )

        return output, aux_loss


class MoEMLP(MoEMLPBase):
    """Mixture of Experts MLP wrapper.

    This class selects the appropriate expert class (MLP or MLPSwiGLU) based on
    the activation configuration.
    """

    def __init__(self, cfg, use_einops=True):
        """Initialize MoE MLP layer.

        Args:
            cfg: Model configuration
            use_einops: If True, use einops implementations, else PyTorch
        """
        # Select expert class based on activation
        # Create a wrapper class that passes use_einops to the expert
        if cfg.activation == Activation.SWIGLU:
            class ExpertWrapper(nn.Module):
                def __init__(self, cfg):
                    super().__init__()
                    self.expert = MLPSwiGLU(cfg, use_einops=use_einops)

                def forward(self, x):
                    return self.expert(x)
            expert_class = ExpertWrapper
        else:
            class ExpertWrapper(nn.Module):
                def __init__(self, cfg):
                    super().__init__()
                    self.expert = MLP(cfg, use_einops=use_einops)

                def forward(self, x):
                    return self.expert(x)
            expert_class = ExpertWrapper

        super().__init__(cfg, expert_class)


# Backward compatibility aliases
MLPWithEinops = MLP
MLPWithoutEinops = MLP
MLPSwiGLUWithEinops = MLPSwiGLU
MLPSwiGLUWithoutEinops = MLPSwiGLU
MoEMLPWithEinops = MoEMLP
MoEMLPWithoutEinops = MoEMLP


def create_moe_mlp_layer(cfg, use_einops=True):
    """Factory function to create MoE MLP layer.

    Args:
        cfg: Model configuration
        use_einops: If True, use einops implementations, else PyTorch

    Returns:
        MoE MLP layer or None if MoE is disabled
    """
    if not cfg.use_moe:
        return None

    return MoEMLP(cfg, use_einops=use_einops)


def create_mlp_layer(cfg, use_einops=True):
    """Factory function to create appropriate MLP layer.

    Args:
        cfg: Model configuration
        use_einops: If True, use einops implementations, else PyTorch

    Returns:
        MLP layer (standard, SwiGLU, or MoE based on config)
    """
    # Check if MoE is enabled
    if cfg.use_moe:
        return create_moe_mlp_layer(cfg, use_einops)

    if cfg.activation == Activation.SWIGLU:
        return MLPSwiGLU(cfg, use_einops=use_einops)
    else:  # GELU
        return MLP(cfg, use_einops=use_einops)
