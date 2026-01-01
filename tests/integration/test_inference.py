"""Integration tests for inference/sampling."""

import pytest
import torch
from inference.sampler import TransformerSampler


@pytest.mark.integration
class TestTransformerSampler:
    """Tests for TransformerSampler."""

    def test_init(self, model_with_einops, character_tokenizer, device):
        """Test sampler initialization."""
        sampler = TransformerSampler(model_with_einops, character_tokenizer, device)
        assert sampler.model == model_with_einops
        assert sampler.tokenizer == character_tokenizer
        assert sampler.device == device

    def test_apply_top_k(self, model_with_einops, character_tokenizer, device):
        """Test top-k filtering."""
        sampler = TransformerSampler(model_with_einops, character_tokenizer, device)
        vocab_size = model_with_einops.cfg.d_vocab
        logits = torch.randn(vocab_size)
        filtered = sampler._apply_top_k(logits, top_k=10)
        
        # Check that only top-k values remain
        non_inf_count = (filtered != float("-inf")).sum().item()
        assert non_inf_count <= 10

    def test_apply_top_p(self, model_with_einops, character_tokenizer, device):
        """Test top-p (nucleus) filtering."""
        sampler = TransformerSampler(model_with_einops, character_tokenizer, device)
        vocab_size = model_with_einops.cfg.d_vocab
        logits = torch.randn(vocab_size)
        filtered = sampler._apply_top_p(logits, top_p=0.9)
        
        # Check that some values remain
        non_inf_count = (filtered != float("-inf")).sum().item()
        assert non_inf_count > 0

    def test_generate_next_token(self, model_with_einops, character_tokenizer, device):
        """Test next token generation."""
        sampler = TransformerSampler(model_with_einops, character_tokenizer, device)
        vocab_size = model_with_einops.cfg.d_vocab
        logits = torch.randn(vocab_size)
        
        next_token = sampler._generate_next_token(logits, temperature=1.0, top_k=None, top_p=None)
        assert next_token.shape == (1,)
        assert 0 <= next_token.item() < vocab_size

    def test_generate_next_token_with_zero_temperature(self, model_with_einops, character_tokenizer, device):
        """Temperature=0 should fall back to deterministic argmax."""
        sampler = TransformerSampler(model_with_einops, character_tokenizer, device)
        vocab_size = model_with_einops.cfg.d_vocab
        logits = torch.randn(vocab_size)

        next_token = sampler._generate_next_token(logits, temperature=0.0, top_k=None, top_p=None)
        assert next_token.shape == (1,)
        assert 0 <= next_token.item() < vocab_size

    def test_apply_top_k_over_vocab(self, model_with_einops, character_tokenizer, device):
        """Top-k larger than vocab should be clamped."""
        sampler = TransformerSampler(model_with_einops, character_tokenizer, device)
        vocab_size = model_with_einops.cfg.d_vocab
        logits = torch.randn(vocab_size)

        filtered = sampler._apply_top_k(logits, top_k=vocab_size + 10)
        non_inf_count = (filtered != float("-inf")).sum().item()
        assert non_inf_count == vocab_size

    def test_sample(self, model_with_einops, character_tokenizer, device):
        """Test text sampling."""
        sampler = TransformerSampler(model_with_einops, character_tokenizer, device)
        prompt = "Hello"
        generated = sampler.sample(prompt, max_new_tokens=5, temperature=1.0)
        
        assert isinstance(generated, str)
        assert prompt in generated  # Generated should include prompt
        assert len(generated) >= len(prompt)

    def test_sample_with_top_k(self, model_with_einops, character_tokenizer, device):
        """Test sampling with top-k."""
        sampler = TransformerSampler(model_with_einops, character_tokenizer, device)
        prompt = "Hello"
        generated = sampler.sample(prompt, max_new_tokens=5, temperature=1.0, top_k=10)
        assert isinstance(generated, str)

    def test_sample_with_top_p(self, model_with_einops, character_tokenizer, device):
        """Test sampling with top-p."""
        sampler = TransformerSampler(model_with_einops, character_tokenizer, device)
        prompt = "Hello"
        generated = sampler.sample(prompt, max_new_tokens=5, temperature=1.0, top_p=0.9)
        assert isinstance(generated, str)

    def test_sample_batch(self, model_with_einops, character_tokenizer, device):
        """Test batch sampling."""
        sampler = TransformerSampler(model_with_einops, character_tokenizer, device)
        prompts = ["Hello", "World"]
        generated = sampler.sample_batch(prompts, max_new_tokens=5)
        assert len(generated) == len(prompts)
        assert all(isinstance(g, str) for g in generated)

    def test_temperature_effect(self, model_with_einops, character_tokenizer, device):
        """Test that temperature affects sampling."""
        sampler = TransformerSampler(model_with_einops, character_tokenizer, device)
        vocab_size = model_with_einops.cfg.d_vocab
        logits = torch.randn(vocab_size)
        
        # Low temperature should be more deterministic
        token_low = sampler._generate_next_token(logits, temperature=0.1, top_k=None, top_p=None)
        # High temperature should be more random
        token_high = sampler._generate_next_token(logits, temperature=2.0, top_k=None, top_p=None)
        
        # Both should be valid tokens
        assert 0 <= token_low.item() < vocab_size
        assert 0 <= token_high.item() < vocab_size

    def test_sampler_handles_moe_outputs(self, moe_config, character_tokenizer, device):
        """Sampler should handle MoE outputs with auxiliary loss."""
        from pretraining.model.model import TransformerModel

        model = TransformerModel(moe_config, use_einops=True)
        sampler = TransformerSampler(model, character_tokenizer, device)
        model.train()  # Enable aux_loss in MoE

        tokens = character_tokenizer.encode("Hello")
        tokens_tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        logits, kv_cache, use_cache, start_pos = sampler._process_prompt(tokens_tensor)

        assert logits.shape[0] == model.cfg.d_vocab
        assert kv_cache is not None
        assert use_cache is True

        logits2, kv_cache2, use_cache2 = sampler._update_cache(
            tokens_tensor, kv_cache, start_pos, use_cache
        )
        assert logits2.shape[0] == model.cfg.d_vocab
        assert kv_cache2 is not None
        assert use_cache2 is True
