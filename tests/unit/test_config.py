"""Unit tests for configuration classes."""

import pytest
from config import ModelConfig, Architecture, PositionalEncoding, Normalization, Activation, RouterType
from pretraining.training.training_args import TransformerTrainingArgs
from finetuning.training.finetuning_args import FinetuningArgs


@pytest.mark.unit
class TestModelConfig:
    """Tests for ModelConfig."""

    def test_init_gpt(self):
        """Test GPT config initialization."""
        cfg = ModelConfig(architecture=Architecture.GPT)
        assert cfg.architecture == Architecture.GPT
        assert cfg.positional_encoding == PositionalEncoding.LEARNED
        assert cfg.normalization == Normalization.LAYERNORM
        assert cfg.activation == Activation.GELU

    def test_init_llama(self):
        """Test LLaMA config initialization."""
        cfg = ModelConfig(architecture=Architecture.LLAMA)
        assert cfg.architecture == Architecture.LLAMA
        assert cfg.positional_encoding == PositionalEncoding.ROPE
        assert cfg.normalization == Normalization.RMSNORM
        assert cfg.activation == Activation.SWIGLU

    def test_init_olmo(self):
        """Test OLMo config initialization."""
        cfg = ModelConfig(architecture=Architecture.OLMO)
        assert cfg.architecture == Architecture.OLMO
        assert cfg.positional_encoding == PositionalEncoding.ALIBI
        assert cfg.normalization == Normalization.LAYERNORM
        assert cfg.activation == Activation.SWIGLU

    def test_gpt_small(self):
        """Test GPT small preset."""
        cfg = ModelConfig.gpt_small()
        assert cfg.architecture == Architecture.GPT
        assert cfg.d_model == 256
        assert cfg.n_heads == 4
        assert cfg.n_layers == 4

    def test_llama_small(self):
        """Test LLaMA small preset."""
        cfg = ModelConfig.llama_small()
        assert cfg.architecture == Architecture.LLAMA
        assert cfg.d_model == 256
        assert cfg.n_heads == 4
        assert cfg.n_layers == 4

    def test_olmo_small(self):
        """Test OLMo small preset."""
        cfg = ModelConfig.olmo_small()
        assert cfg.architecture == Architecture.OLMO
        assert cfg.d_model == 256
        assert cfg.n_heads == 4
        assert cfg.n_layers == 4

    def test_to_dict(self):
        """Test config to dict conversion."""
        cfg = ModelConfig.gpt_small()
        cfg_dict = cfg.to_dict()
        assert isinstance(cfg_dict, dict)
        assert cfg_dict['architecture'] == 'gpt'
        assert cfg_dict['d_model'] == 256

    def test_from_dict(self):
        """Test config from dict reconstruction."""
        cfg = ModelConfig.gpt_small()
        cfg_dict = cfg.to_dict()
        cfg_reconstructed = ModelConfig.from_dict(cfg_dict)
        assert cfg_reconstructed.architecture == cfg.architecture
        assert cfg_reconstructed.d_model == cfg.d_model

    def test_moe_config(self):
        """Test MoE configuration."""
        cfg = ModelConfig(architecture=Architecture.LLAMA, use_moe=True)
        assert cfg.use_moe
        assert cfg.num_experts == 8
        assert cfg.num_experts_per_tok == 2
        assert cfg.router_type == RouterType.TOP_K

    def test_moe_with_shared_experts(self):
        """Test MoE with shared experts."""
        cfg = ModelConfig(
            architecture=Architecture.LLAMA,
            use_moe=True,
            use_shared_experts=True
        )
        assert cfg.use_shared_experts
        assert cfg.router_type == RouterType.TOP_K_WITH_SHARED

    def test_gqa_config(self):
        """Test GQA configuration."""
        cfg = ModelConfig(architecture=Architecture.LLAMA, n_heads=8, n_kv_heads=2)
        assert cfg.n_heads == 8
        assert cfg.n_kv_heads == 2
        assert cfg.n_heads % cfg.n_kv_heads == 0

    def test_invalid_n_kv_heads(self):
        """Test invalid n_kv_heads raises error."""
        with pytest.raises(ValueError):
            ModelConfig(architecture=Architecture.LLAMA, n_heads=8, n_kv_heads=3)  # Not divisible

    def test_n_kv_heads_greater_than_n_heads(self):
        """Test n_kv_heads > n_heads raises error."""
        with pytest.raises(ValueError):
            ModelConfig(architecture=Architecture.LLAMA, n_heads=4, n_kv_heads=8)

    def test_rope_requires_even_d_head(self):
        """Test RoPE requires even d_head."""
        with pytest.raises(ValueError):
            ModelConfig(architecture=Architecture.LLAMA, d_head=63)


@pytest.mark.unit
class TestTransformerTrainingArgs:
    """Tests for TransformerTrainingArgs."""

    def test_init(self):
        """Test initialization."""
        args = TransformerTrainingArgs()
        assert args.batch_size == 32
        assert args.epochs == 10
        assert args.lr == 1e-3

    def test_custom_args(self):
        """Test custom arguments."""
        args = TransformerTrainingArgs(
            batch_size=16,
            epochs=5,
            lr=1e-4
        )
        assert args.batch_size == 16
        assert args.epochs == 5
        assert args.lr == 1e-4


@pytest.mark.unit
class TestFinetuningArgs:
    """Tests for FinetuningArgs."""

    def test_init(self):
        """Test initialization."""
        args = FinetuningArgs()
        assert args.batch_size == 4
        assert args.epochs == 3
        assert args.lr == 1e-5  # Lower than pre-training
        assert args.use_lora == False

    def test_lora_args(self):
        """Test LoRA arguments."""
        args = FinetuningArgs(
            use_lora=True,
            lora_rank=16,
            lora_alpha=16.0
        )
        assert args.use_lora
        assert args.lora_rank == 16
        assert args.lora_alpha == 16.0
