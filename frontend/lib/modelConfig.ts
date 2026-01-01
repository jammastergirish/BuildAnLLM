export type ModelConfig = {
  d_model: number;
  n_heads: number;
  n_layers: number;
  n_ctx: number;
  d_head: number;
  d_mlp: number;
  positional_encoding: "learned" | "rope" | "alibi" | "none";
  normalization: "layernorm" | "rmsnorm";
  activation: "gelu" | "swiglu";
  n_kv_heads?: number;
  rope_theta?: number;
  use_moe?: boolean;
  num_experts?: number;
  num_experts_per_tok?: number;
  use_shared_experts?: boolean;
  num_shared_experts?: number;
  router_type?: "top_k" | "top_k_with_shared";
  load_balancing_loss_weight?: number;
  expert_capacity_factor?: number;
  tokenizer_type?: string;
  use_einops?: boolean;
};

export const defaultModelConfig: ModelConfig = {
  positional_encoding: "learned",
  normalization: "layernorm",
  activation: "gelu",
  d_model: 256,
  n_heads: 4,
  n_layers: 4,
  n_ctx: 256,
  d_head: 64,
  d_mlp: 1024,
  n_kv_heads: 4,
  rope_theta: 10000,
  use_moe: false,
  num_experts: 8,
  num_experts_per_tok: 2,
  use_shared_experts: false,
  num_shared_experts: 2,
  router_type: "top_k",
  load_balancing_loss_weight: 0.01,
  expert_capacity_factor: 1.25,
};

const sizePresets = {
  small: {
    d_model: 256,
    n_heads: 4,
    n_layers: 4,
    n_ctx: 256,
    d_head: 64,
    d_mlp: 1024,
  },
  medium: {
    d_model: 512,
    n_heads: 8,
    n_layers: 6,
    n_ctx: 512,
    d_head: 64,
    d_mlp: 2048,
  },
  full: {
    d_model: 768,
    n_heads: 12,
    n_layers: 12,
    n_ctx: 1024,
    d_head: 64,
    d_mlp: 3072,
  },
};

export type ModelSize = keyof typeof sizePresets;

export function applySizePreset(config: ModelConfig, size: ModelSize): ModelConfig {
  return { ...config, ...sizePresets[size], n_kv_heads: sizePresets[size].n_heads };
}

export function applyPreset(config: ModelConfig, preset: string): ModelConfig {
  const base = { ...config };
  switch (preset) {
    case "gpt":
      return {
        ...base,
        positional_encoding: "learned",
        normalization: "layernorm",
        activation: "gelu",
        use_moe: false,
      };
    case "llama":
      return {
        ...base,
        positional_encoding: "rope",
        normalization: "rmsnorm",
        activation: "swiglu",
        use_moe: true,
        num_experts: 16,
        num_experts_per_tok: 2,
        use_shared_experts: true,
        num_shared_experts: 2,
        router_type: "top_k_with_shared",
        n_kv_heads: Math.max(1, Math.floor(base.n_heads / 4)),
      };
    case "olmo":
      return {
        ...base,
        positional_encoding: "alibi",
        normalization: "layernorm",
        activation: "swiglu",
        use_moe: false,
      };
    case "deepseek":
      return {
        ...base,
        positional_encoding: "rope",
        normalization: "rmsnorm",
        activation: "swiglu",
        use_moe: true,
        num_experts: 64,
        num_experts_per_tok: 6,
        use_shared_experts: true,
        num_shared_experts: 2,
        router_type: "top_k_with_shared",
        n_kv_heads: Math.max(1, Math.floor(base.n_heads / 4)),
      };
    case "mixtral":
      return {
        ...base,
        positional_encoding: "rope",
        normalization: "rmsnorm",
        activation: "swiglu",
        use_moe: true,
        num_experts: 8,
        num_experts_per_tok: 2,
        use_shared_experts: false,
        router_type: "top_k",
        n_kv_heads: Math.max(1, Math.floor(base.n_heads / 4)),
      };
    default:
      return base;
  }
}

export function estimateParams(config: ModelConfig) {
  const attnParams = config.n_layers * 4 * (config.d_model * config.d_model);
  const mlpParams = config.n_layers * 2 * (config.d_model * config.d_mlp);
  const embedParams = config.d_model * 10000;
  return (attnParams + mlpParams + embedParams) / 1e6;
}
