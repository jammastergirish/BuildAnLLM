"""Comprehensive dimension and shape tests for all components."""

import pytest
import torch
from config import ModelConfig
from pretraining.model.model import TransformerModel
from pretraining.attention.attention import Attention
from pretraining.mlp.mlp import MLP, MLPSwiGLU
from pretraining.embeddings.embed import EmbedWithoutTorch as Embed, UnembedWithoutTorch as Unembed
from pretraining.normalization.layernorm import LayerNorm
from pretraining.normalization.rmsnorm import RMSNorm
from pretraining.positional_embeddings.positional_embedding import PosEmbed
from pretraining.positional_embeddings.rope import RoPE
from pretraining.positional_embeddings.alibi import ALiBi
from pretraining.transformer_blocks.transformer_block import TransformerBlock
from finetuning.peft.lora_wrappers import (
    create_lora_matrices,
    apply_lora_to_attention,
    apply_lora_to_mlp,
    einsum_with_lora,
)
from finetuning.peft.lora_utils import convert_model_to_lora


@pytest.mark.unit
class TestEmbeddingDimensions:
    """Tests for embedding dimension correctness."""

    def test_embed_dimensions(self, small_config):
        """Test that embeddings produce correct dimensions."""
        embed = Embed(small_config)
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, small_config.d_vocab, (batch_size, seq_len))

        output = embed(tokens)
        assert output.shape == (batch_size, seq_len, small_config.d_model)

    def test_unembed_dimensions(self, small_config):
        """Test that unembeddings produce correct dimensions."""
        unembed = Unembed(small_config)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)

        logits = unembed(residual)
        assert logits.shape == (batch_size, seq_len, small_config.d_vocab)

    def test_embed_weight_dimensions(self, small_config):
        """Test that embedding weights have correct dimensions."""
        embed = Embed(small_config)
        assert embed.W_E.shape == (small_config.d_vocab, small_config.d_model)

    def test_unembed_weight_dimensions(self, small_config):
        """Test that unembedding weights have correct dimensions."""
        unembed = Unembed(small_config)
        # Unembed weight is (d_model, d_vocab) - transposed from embedding
        assert unembed.W_U.shape == (
            small_config.d_model, small_config.d_vocab)


@pytest.mark.unit
class TestNormalizationDimensions:
    """Tests for normalization layer dimensions."""

    def test_layernorm_dimensions(self, small_config):
        """Test LayerNorm preserves dimensions."""
        ln = LayerNorm(small_config, use_einops=True)
        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, small_config.d_model)

        output = ln(x)
        assert output.shape == x.shape

    def test_rmsnorm_dimensions(self, llama_config):
        """Test RMSNorm preserves dimensions."""
        rms = RMSNorm(llama_config, use_einops=True)
        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, llama_config.d_model)

        output = rms(x)
        assert output.shape == x.shape

    def test_layernorm_weight_dimensions(self, small_config):
        """Test LayerNorm weight dimensions."""
        ln = LayerNorm(small_config, use_einops=True)
        assert ln.w.shape == (small_config.d_model,)
        assert ln.b.shape == (small_config.d_model,)

    def test_rmsnorm_weight_dimensions(self, llama_config):
        """Test RMSNorm weight dimensions."""
        rms = RMSNorm(llama_config, use_einops=True)
        assert rms.w.shape == (llama_config.d_model,)


@pytest.mark.unit
class TestPositionalEncodingDimensions:
    """Tests for positional encoding dimensions."""

    def test_pos_embed_dimensions(self, gpt_config):
        """Test learned positional embeddings dimensions."""
        pos_embed = PosEmbed(gpt_config, use_einops=True)
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, gpt_config.d_vocab, (batch_size, seq_len))

        # PosEmbed takes tokens, not residual
        output = pos_embed(tokens)
        assert output.shape == (batch_size, seq_len, gpt_config.d_model)

    def test_pos_embed_weight_dimensions(self, gpt_config):
        """Test learned positional embedding weight dimensions."""
        pos_embed = PosEmbed(gpt_config, use_einops=True)
        assert pos_embed.W_pos.shape == (gpt_config.n_ctx, gpt_config.d_model)

    def test_rope_dimensions(self, llama_config):
        """Test RoPE preserves dimensions."""
        rope = RoPE(llama_config)
        batch_size, seq_len, n_heads, d_head = 2, 5, llama_config.n_heads, llama_config.d_head
        n_kv_heads = llama_config.n_kv_heads
        q = torch.randn(batch_size, seq_len, n_heads, d_head)
        k = torch.randn(batch_size, seq_len, n_kv_heads, d_head)
        positions = torch.arange(seq_len)

        q_rotated, k_rotated = rope(q, k, positions)
        assert q_rotated.shape == q.shape
        assert k_rotated.shape == k.shape

    def test_alibi_bias_dimensions(self, olmo_config):
        """Test ALiBi bias dimensions."""
        alibi = ALiBi(olmo_config)
        batch_size, seq_len = 2, 5
        device = torch.device('cpu')

        # ALiBi bias matrix should be [n_heads, seq_len, seq_len]
        bias = alibi.get_bias(seq_len, device)
        assert bias.shape == (olmo_config.n_heads, seq_len, seq_len)


@pytest.mark.unit
class TestAttentionDimensions:
    """Comprehensive tests for attention dimensions."""

    def test_attention_weight_dimensions(self, small_config):
        """Test attention weight matrix dimensions."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=True)

        assert attn.W_Q.shape == (
            small_config.n_heads, small_config.d_head, small_config.d_model)
        assert attn.W_K.shape == (
            small_config.n_kv_heads, small_config.d_head, small_config.d_model)
        assert attn.W_V.shape == (
            small_config.n_kv_heads, small_config.d_head, small_config.d_model)
        assert attn.W_O.shape == (
            small_config.n_heads, small_config.d_head, small_config.d_model)

    def test_qkv_computation_dimensions(self, small_config):
        """Test QKV computation produces correct dimensions."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)

        q, k, v = attn._compute_qkv(residual)
        assert q.shape == (batch_size, seq_len,
                           small_config.n_heads, small_config.d_head)
        assert k.shape == (batch_size, seq_len,
                           small_config.n_kv_heads, small_config.d_head)
        assert v.shape == (batch_size, seq_len,
                           small_config.n_kv_heads, small_config.d_head)

    def test_attention_scores_dimensions(self, small_config):
        """Test attention scores have correct dimensions."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=True)
        batch_size, seq_len = 2, 5
        q = torch.randn(batch_size, seq_len,
                        small_config.n_heads, small_config.d_head)
        k = torch.randn(batch_size, seq_len,
                        small_config.n_heads, small_config.d_head)

        scores = attn._compute_attention_scores(q, k)
        assert scores.shape == (
            batch_size, small_config.n_heads, seq_len, seq_len)

    def test_attention_output_dimensions(self, small_config):
        """Test attention output dimensions."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)

        output, cache = attn(residual)
        assert output.shape == (batch_size, seq_len, small_config.d_model)
        assert len(cache) == 2
        assert cache[0].shape == (
            batch_size, seq_len, small_config.n_kv_heads, small_config.d_head)
        assert cache[1].shape == (
            batch_size, seq_len, small_config.n_kv_heads, small_config.d_head)

    def test_attention_cache_dimensions(self, small_config):
        """Test attention cache accumulation dimensions."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=True)
        batch_size = 2

        # First forward pass
        residual1 = torch.randn(batch_size, 5, small_config.d_model)
        output1, cache1 = attn(residual1, cache=None, start_pos=0)

        # Second forward pass with cache
        residual2 = torch.randn(batch_size, 3, small_config.d_model)
        output2, cache2 = attn(residual2, cache=cache1, start_pos=5)

        assert output2.shape == (batch_size, 3, small_config.d_model)
        assert cache2[0].shape[1] == 8  # 5 + 3
        assert cache2[1].shape[1] == 8

    def test_gqa_dimensions(self, gqa_config):
        """Test Grouped Query Attention dimensions."""
        attn = Attention(gqa_config, rope=None, alibi=None, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, gqa_config.d_model)

        q, k, v = attn._compute_qkv(residual)
        assert q.shape == (batch_size, seq_len,
                           gqa_config.n_heads, gqa_config.d_head)
        assert k.shape == (batch_size, seq_len,
                           gqa_config.n_kv_heads, gqa_config.d_head)
        assert v.shape == (batch_size, seq_len,
                           gqa_config.n_kv_heads, gqa_config.d_head)

        output, cache = attn(residual)
        assert output.shape == (batch_size, seq_len, gqa_config.d_model)
        assert cache[0].shape[2] == gqa_config.n_kv_heads


@pytest.mark.unit
class TestMLPDimensions:
    """Comprehensive tests for MLP dimensions."""

    def test_mlp_weight_dimensions(self, small_config):
        """Test MLP weight matrix dimensions."""
        mlp = MLP(small_config, use_einops=True)

        # MLP weights are (d_model, d_mlp) for W_in and (d_mlp, d_model) for W_out
        assert mlp.W_in.shape == (small_config.d_model, small_config.d_mlp)
        assert mlp.W_out.shape == (small_config.d_mlp, small_config.d_model)

    def test_mlp_forward_dimensions(self, small_config):
        """Test MLP forward pass dimensions."""
        mlp = MLP(small_config, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)

        output = mlp(residual)
        assert output.shape == residual.shape

    def test_swiglu_weight_dimensions(self, llama_config):
        """Test SwiGLU MLP weight dimensions."""
        mlp = MLPSwiGLU(llama_config, use_einops=True)

        # SwiGLU weights are (d_model, d_mlp) for W_gate/W_up and (d_mlp, d_model) for W_out
        assert mlp.W_gate.shape == (llama_config.d_model, llama_config.d_mlp)
        assert mlp.W_up.shape == (llama_config.d_model, llama_config.d_mlp)
        assert mlp.W_out.shape == (llama_config.d_mlp, llama_config.d_model)

    def test_swiglu_forward_dimensions(self, llama_config):
        """Test SwiGLU forward pass dimensions."""
        mlp = MLPSwiGLU(llama_config, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, llama_config.d_model)

        output = mlp(residual)
        assert output.shape == residual.shape

    def test_mlp_intermediate_dimensions(self, small_config):
        """Test MLP intermediate activations have correct dimensions."""
        mlp = MLP(small_config, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)

        # Check intermediate dimensions - W_in is (d_model, d_mlp), so residual @ W_in gives (batch, seq, d_mlp)
        pre_act = residual @ mlp.W_in
        assert pre_act.shape == (batch_size, seq_len, small_config.d_mlp)


@pytest.mark.unit
class TestTransformerBlockDimensions:
    """Tests for transformer block dimensions."""

    def test_transformer_block_dimensions(self, small_config):
        """Test transformer block preserves dimensions."""
        block = TransformerBlock(
            small_config, rope=None, alibi=None, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)

        output, cache, aux_loss = block(residual, cache=None, start_pos=0)
        assert output.shape == residual.shape
        assert len(cache) == 2  # K and V cache
        assert aux_loss is None or isinstance(aux_loss, torch.Tensor)

    def test_transformer_block_cache_dimensions(self, small_config):
        """Test transformer block cache dimensions."""
        block = TransformerBlock(
            small_config, rope=None, alibi=None, use_einops=True)
        batch_size = 2

        # First forward pass
        residual1 = torch.randn(batch_size, 5, small_config.d_model)
        output1, cache1, _ = block(residual1, cache=None, start_pos=0)

        # Second forward pass with cache
        residual2 = torch.randn(batch_size, 3, small_config.d_model)
        output2, cache2, _ = block(residual2, cache=cache1, start_pos=5)

        assert output2.shape == (batch_size, 3, small_config.d_model)
        assert cache2[0].shape[1] == 8  # Accumulated sequence length


@pytest.mark.unit
class TestModelDimensions:
    """Comprehensive tests for full model dimensions."""

    def test_model_forward_dimensions(self, small_config, sample_tokens):
        """Test model forward pass produces correct dimensions."""
        model = TransformerModel(small_config, use_einops=True)
        batch_size, seq_len = sample_tokens.shape

        result = model(sample_tokens)
        logits, cache = result

        assert logits.shape == (batch_size, seq_len, small_config.d_vocab)
        assert isinstance(cache, list)
        assert len(cache) == len(model.blocks)

    def test_model_cache_dimensions(self, small_config):
        """Test model cache dimensions."""
        model = TransformerModel(small_config, use_einops=True)
        batch_size = 2

        # First forward pass
        tokens1 = torch.randint(0, small_config.d_vocab, (batch_size, 5))
        logits1, cache1 = model(tokens1, cache=None, start_pos=0)

        # Second forward pass with cache
        tokens2 = torch.randint(0, small_config.d_vocab, (batch_size, 3))
        logits2, cache2 = model(tokens2, cache=cache1, start_pos=5)

        assert logits2.shape == (batch_size, 3, small_config.d_vocab)
        # Cache should accumulate
        assert cache2[0][0].shape[1] == 8  # 5 + 3

    def test_model_different_batch_sizes(self, small_config):
        """Test model handles different batch sizes correctly."""
        model = TransformerModel(small_config, use_einops=True)

        for batch_size in [1, 2, 4, 8]:
            tokens = torch.randint(0, small_config.d_vocab, (batch_size, 5))
            logits, _ = model(tokens)
            assert logits.shape[0] == batch_size

    def test_model_different_sequence_lengths(self, small_config):
        """Test model handles different sequence lengths correctly."""
        model = TransformerModel(small_config, use_einops=True)
        batch_size = 2

        for seq_len in [1, 5, 10, 20]:
            tokens = torch.randint(
                0, small_config.d_vocab, (batch_size, seq_len))
            logits, _ = model(tokens)
            assert logits.shape[1] == seq_len
            assert logits.shape[2] == small_config.d_vocab

    def test_model_all_components_dimensions(self, small_config):
        """Test that all model components maintain correct dimensions."""
        model = TransformerModel(small_config, use_einops=True)
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, small_config.d_vocab, (batch_size, seq_len))

        # Check embedding dimensions
        embedded = model.embed(tokens)
        assert embedded.shape == (batch_size, seq_len, small_config.d_model)

        # Check block dimensions
        residual = embedded
        for block in model.blocks:
            residual, _, _ = block(residual, cache=None, start_pos=0)
            assert residual.shape == (
                batch_size, seq_len, small_config.d_model)

        # Check final normalization
        normalized = model.ln_f(residual)
        assert normalized.shape == (batch_size, seq_len, small_config.d_model)

        # Check unembedding
        logits = model.unembed(normalized)
        assert logits.shape == (batch_size, seq_len, small_config.d_vocab)


@pytest.mark.unit
class TestDimensionConsistency:
    """Tests for dimension consistency across operations."""

    def test_attention_mlp_residual_dimensions(self, small_config):
        """Test that attention and MLP maintain residual dimensions."""
        block = TransformerBlock(
            small_config, rope=None, alibi=None, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)

        # Attention should preserve dimensions
        attn_output, _ = block.attn(
            block.ln1(residual), cache=None, start_pos=0)
        assert attn_output.shape == (batch_size, seq_len, small_config.d_model)

        # Residual connection should preserve dimensions
        residual_after_attn = residual + attn_output
        assert residual_after_attn.shape == residual.shape

        # MLP should preserve dimensions
        mlp_output = block.mlp(block.ln2(residual_after_attn))
        assert mlp_output.shape == (batch_size, seq_len, small_config.d_model)

        # Final residual should preserve dimensions
        final_residual = residual_after_attn + mlp_output
        assert final_residual.shape == residual.shape

    def test_einops_vs_non_einops_dimensions(self, small_config):
        """Test that einops and non-einops produce same dimensions."""
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, small_config.d_model)

        attn_einops = Attention(small_config, rope=None,
                                alibi=None, use_einops=True)
        attn_no_einops = Attention(
            small_config, rope=None, alibi=None, use_einops=False)

        output_einops, cache_einops = attn_einops(residual)
        output_no_einops, cache_no_einops = attn_no_einops(residual)

        assert output_einops.shape == output_no_einops.shape
        assert cache_einops[0].shape == cache_no_einops[0].shape
        assert cache_einops[1].shape == cache_no_einops[1].shape


@pytest.mark.unit
class TestDimensionErrors:
    """Tests that catch dimension mismatch errors."""

    def test_wrong_input_dimensions_raise_error(self, small_config):
        """Test that wrong input dimensions raise errors."""
        attn = Attention(small_config, rope=None, alibi=None, use_einops=True)

        # Wrong d_model dimension
        wrong_residual = torch.randn(2, 5, small_config.d_model + 10)
        with pytest.raises((RuntimeError, AssertionError)):
            attn(wrong_residual)

    def test_wrong_batch_dimension_raises_error(self, small_config):
        """Test that mismatched batch dimensions raise errors."""
        model = TransformerModel(small_config, use_einops=True)

        # Different batch sizes in cache
        tokens1 = torch.randint(0, small_config.d_vocab, (2, 5))
        logits1, cache1 = model(tokens1, cache=None, start_pos=0)

        # Try to use cache with different batch size
        tokens2 = torch.randint(0, small_config.d_vocab, (3, 1))
        with pytest.raises((RuntimeError, AssertionError)):
            model(tokens2, cache=cache1, start_pos=5)


@pytest.mark.unit
class TestLoraMatrixDimensions:
    """Tests for LoRA matrix dimension correctness."""

    def test_2d_lora_matrix_dimensions(self):
        """Test that 2D LoRA matrices have correct dimensions for matrix multiplication."""
        out_dim, in_dim = 128, 256
        rank = 8
        weight_shape = (out_dim, in_dim)

        lora_A, lora_B, scaling, _ = create_lora_matrices(
            weight_shape, rank, alpha=8.0, dropout=0.0)

        # Check shapes
        assert lora_A.shape == (
            rank, in_dim), f"lora_A should be [rank, in_dim], got {lora_A.shape}"
        assert lora_B.shape == (
            out_dim, rank), f"lora_B should be [out_dim, rank], got {lora_B.shape}"

        # Verify matrix multiplication compatibility: B @ A should produce [out_dim, in_dim]
        lora_weight = torch.matmul(lora_B, lora_A)
        assert lora_weight.shape == (out_dim, in_dim), \
            f"B @ A should be [out_dim, in_dim], got {lora_weight.shape}"

    def test_3d_lora_matrix_dimensions(self):
        """Test that 3D LoRA matrices have correct dimensions for batch matrix multiplication."""
        n_heads, d_head, d_model = 4, 64, 256
        rank = 8
        weight_shape = (n_heads, d_head, d_model)

        lora_A, lora_B, scaling, _ = create_lora_matrices(
            weight_shape, rank, alpha=8.0, dropout=0.0)

        # Check shapes
        assert lora_A.shape == (n_heads, rank, d_model), \
            f"lora_A should be [n_heads, rank, d_model], got {lora_A.shape}"
        assert lora_B.shape == (n_heads, d_head, rank), \
            f"lora_B should be [n_heads, d_head, rank], got {lora_B.shape}"

        # Verify batch matrix multiplication compatibility: B @ A should produce [n_heads, d_head, d_model]
        lora_weight = torch.bmm(lora_B, lora_A)
        assert lora_weight.shape == (n_heads, d_head, d_model), \
            f"B @ A should be [n_heads, d_head, d_model], got {lora_weight.shape}"

    def test_attention_lora_matrix_dimensions(self, llama_model_with_einops):
        """Test that attention LoRA matrices match base weight dimensions."""
        attn = llama_model_with_einops.blocks[0].attn
        apply_lora_to_attention(attn, rank=8, alpha=8.0, dropout=0.0)

        for param_name in ['W_Q', 'W_K', 'W_V', 'W_O']:
            base_weight = getattr(attn, param_name)
            lora_A = getattr(attn, f'{param_name}_lora_A')
            lora_B = getattr(attn, f'{param_name}_lora_B')

            # Base weight is [n_heads, d_head, d_model]
            n_heads, d_head, d_model = base_weight.shape

            # Check LoRA matrix shapes
            assert lora_A.shape == (n_heads, 8, d_model), \
                f"{param_name}_lora_A should be [n_heads, rank, d_model], got {lora_A.shape}"
            assert lora_B.shape == (n_heads, d_head, 8), \
                f"{param_name}_lora_B should be [n_heads, d_head, rank], got {lora_B.shape}"

            # Verify matrix multiplication produces correct shape
            lora_weight = torch.bmm(lora_B, lora_A)
            assert lora_weight.shape == base_weight.shape, \
                f"{param_name} LoRA weight should match base weight shape {base_weight.shape}, got {lora_weight.shape}"

    def test_mlp_lora_matrix_dimensions_swiglu(self, llama_model_with_einops):
        """Test that SwiGLU MLP LoRA matrices match base weight dimensions."""
        mlp = llama_model_with_einops.blocks[0].mlp
        apply_lora_to_mlp(mlp, rank=8, alpha=8.0, dropout=0.0)

        for param_name in ['W_gate', 'W_up', 'W_out']:
            base_weight = getattr(mlp, param_name)
            lora_A = getattr(mlp, f'{param_name}_lora_A')
            lora_B = getattr(mlp, f'{param_name}_lora_B')

            # Base weight is [out_dim, in_dim] for MLP
            out_dim, in_dim = base_weight.shape

            # Check LoRA matrix shapes
            assert lora_A.shape == (8, in_dim), \
                f"{param_name}_lora_A should be [rank, in_dim], got {lora_A.shape}"
            assert lora_B.shape == (out_dim, 8), \
                f"{param_name}_lora_B should be [out_dim, rank], got {lora_B.shape}"

            # Verify matrix multiplication produces correct shape
            lora_weight = torch.matmul(lora_B, lora_A)
            assert lora_weight.shape == base_weight.shape, \
                f"{param_name} LoRA weight should match base weight shape {base_weight.shape}, got {lora_weight.shape}"

    def test_mlp_lora_matrix_dimensions_standard(self, model_with_einops):
        """Test that standard MLP LoRA matrices match base weight dimensions."""
        mlp = model_with_einops.blocks[0].mlp
        apply_lora_to_mlp(mlp, rank=8, alpha=8.0, dropout=0.0)

        for param_name in ['W_in', 'W_out']:
            base_weight = getattr(mlp, param_name)
            lora_A = getattr(mlp, f'{param_name}_lora_A')
            lora_B = getattr(mlp, f'{param_name}_lora_B')

            # Base weight is [out_dim, in_dim] for MLP
            out_dim, in_dim = base_weight.shape

            # Check LoRA matrix shapes
            assert lora_A.shape == (8, in_dim), \
                f"{param_name}_lora_A should be [rank, in_dim], got {lora_A.shape}"
            assert lora_B.shape == (out_dim, 8), \
                f"{param_name}_lora_B should be [out_dim, rank], got {lora_B.shape}"

            # Verify matrix multiplication produces correct shape
            lora_weight = torch.matmul(lora_B, lora_A)
            assert lora_weight.shape == base_weight.shape, \
                f"{param_name} LoRA weight should match base weight shape {base_weight.shape}, got {lora_weight.shape}"


@pytest.mark.unit
class TestLoraEinsumDimensions:
    """Tests for einsum dimension compatibility with LoRA."""

    def test_einsum_2d_dimension_compatibility(self):
        """Test that 2D einsum with LoRA produces correct dimensions."""
        batch_size, seq_len, d_model = 2, 5, 256
        out_dim = 128
        rank = 8

        x = torch.randn(batch_size, seq_len, d_model)
        weight = torch.randn(out_dim, d_model)
        lora_A = torch.randn(rank, d_model)
        lora_B = torch.randn(out_dim, rank)

        pattern = "batch posn d_model, out_dim d_model -> batch posn out_dim"

        # Compute LoRA weight
        lora_weight = torch.matmul(lora_B, lora_A)
        assert lora_weight.shape == (out_dim, d_model)

        # Test einsum
        output = einsum_with_lora(
            x, weight, pattern, lora_A, lora_B, scaling=1.0, dropout_layer=torch.nn.Identity())
        assert output.shape == (batch_size, seq_len, out_dim)

    def test_einsum_3d_dimension_compatibility(self):
        """Test that 3D einsum with LoRA produces correct dimensions."""
        batch_size, seq_len, d_model = 2, 5, 256
        n_heads, d_head = 4, 64
        rank = 8

        x = torch.randn(batch_size, seq_len, d_model)
        weight = torch.randn(n_heads, d_head, d_model)
        lora_A = torch.randn(n_heads, rank, d_model)
        lora_B = torch.randn(n_heads, d_head, rank)

        pattern = "batch posn d_model, n_heads d_head d_model -> batch posn n_heads d_head"

        # Compute LoRA weight
        lora_weight = torch.bmm(lora_B, lora_A)
        assert lora_weight.shape == (n_heads, d_head, d_model)

        # Test einsum
        output = einsum_with_lora(
            x, weight, pattern, lora_A, lora_B, scaling=1.0, dropout_layer=torch.nn.Identity())
        assert output.shape == (batch_size, seq_len, n_heads, d_head)

    def test_einsum_attention_qkv_dimensions(self, llama_model_with_einops):
        """Test that attention Q/K/V einsum with LoRA produces correct dimensions."""
        import einops
        model_lora = convert_model_to_lora(
            llama_model_with_einops, rank=8, alpha=8.0, dropout=0.0)
        attn = model_lora.blocks[0].attn

        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len,
                               llama_model_with_einops.cfg.d_model)

        # Test Q computation using einops (matches actual implementation)
        q = einops.einsum(
            residual, attn.W_Q,
            "batch posn d_model, n_heads d_head d_model -> batch posn n_heads d_head"
        )
        assert q.shape == (
            batch_size, seq_len, llama_model_with_einops.cfg.n_heads, llama_model_with_einops.cfg.d_head)

        # Test LoRA Q computation
        lora_weight_q = torch.bmm(attn.W_Q_lora_B, attn.W_Q_lora_A)
        assert lora_weight_q.shape == attn.W_Q.shape

        q_lora = einops.einsum(
            residual, lora_weight_q,
            "batch posn d_model, n_heads d_head d_model -> batch posn n_heads d_head"
        )
        assert q_lora.shape == q.shape


@pytest.mark.unit
class TestLoraShapeConsistency:
    """Tests for LoRA shape consistency across different configurations."""

    def test_different_ranks_produce_correct_shapes(self):
        """Test that different ranks produce correctly sized LoRA matrices."""
        weight_shape = (128, 256)

        for rank in [1, 4, 8, 16, 32]:
            lora_A, lora_B, _, _ = create_lora_matrices(
                weight_shape, rank, alpha=8.0, dropout=0.0)

            assert lora_A.shape == (
                rank, 256), f"Rank {rank}: lora_A shape incorrect"
            assert lora_B.shape == (
                128, rank), f"Rank {rank}: lora_B shape incorrect"

            # Verify matrix multiplication
            lora_weight = torch.matmul(lora_B, lora_A)
            assert lora_weight.shape == weight_shape, f"Rank {rank}: LoRA weight shape incorrect"

    def test_different_weight_shapes(self):
        """Test LoRA matrices for various weight shapes."""
        test_cases = [
            ((64, 128), 8),      # Small MLP
            ((256, 512), 8),     # Medium MLP
            ((1024, 2048), 8),   # Large MLP
            ((4, 64, 256), 8),  # Small attention
            ((8, 128, 512), 8),  # Medium attention
            ((16, 64, 1024), 8),  # Large attention
        ]

        for weight_shape, rank in test_cases:
            lora_A, lora_B, _, _ = create_lora_matrices(
                weight_shape, rank, alpha=8.0, dropout=0.0)

            if len(weight_shape) == 2:
                out_dim, in_dim = weight_shape
                assert lora_A.shape == (rank, in_dim)
                assert lora_B.shape == (out_dim, rank)
                lora_weight = torch.matmul(lora_B, lora_A)
            else:
                n_heads, d_head, d_model = weight_shape
                assert lora_A.shape == (n_heads, rank, d_model)
                assert lora_B.shape == (n_heads, d_head, rank)
                lora_weight = torch.bmm(lora_B, lora_A)

            assert lora_weight.shape == weight_shape, \
                f"Weight shape {weight_shape}: LoRA weight shape {lora_weight.shape} doesn't match"

    def test_model_forward_shapes_with_lora(self, model_with_einops):
        """Test that model forward pass produces correct shapes with LoRA."""
        model_lora = convert_model_to_lora(
            model_with_einops, rank=8, alpha=8.0, dropout=0.0)

        batch_sizes = [1, 2, 4]
        seq_lens = [1, 5, 10]

        for batch_size in batch_sizes:
            for seq_len in seq_lens:
                tokens = torch.randint(
                    0, model_with_einops.cfg.d_vocab, (batch_size, seq_len))
                result = model_lora(tokens)
                logits, cache = result

                assert logits.shape == (batch_size, seq_len, model_with_einops.cfg.d_vocab), \
                    f"Batch {batch_size}, Seq {seq_len}: logits shape {logits.shape} incorrect"
                assert len(cache) == len(model_lora.blocks), \
                    f"Batch {batch_size}, Seq {seq_len}: cache length incorrect"


@pytest.mark.unit
class TestLoraDimensionErrors:
    """Tests that catch LoRA dimension mismatch errors."""

    def test_incompatible_lora_matrices_raise_error(self):
        """Test that incompatible LoRA matrices raise errors in einsum."""
        x = torch.randn(2, 5, 256)
        weight = torch.randn(128, 256)

        # Wrong rank dimension
        lora_A = torch.randn(8, 256)  # Correct
        lora_B = torch.randn(128, 16)  # Wrong rank (16 instead of 8)

        pattern = "batch posn d_model, out_dim d_model -> batch posn out_dim"

        # This should fail when computing B @ A
        with pytest.raises(RuntimeError):
            # Shape mismatch: [128, 16] @ [8, 256]
            lora_weight = torch.matmul(lora_B, lora_A)

    def test_wrong_input_dimensions_raise_error_lora(self):
        """Test that wrong input dimensions raise errors in LoRA einsum."""
        x = torch.randn(2, 5, 128)  # Wrong d_model (128 instead of 256)
        weight = torch.randn(128, 256)
        lora_A = torch.randn(8, 256)
        lora_B = torch.randn(128, 8)

        pattern = "batch posn d_model, out_dim d_model -> batch posn out_dim"

        # This should fail in einsum due to dimension mismatch
        with pytest.raises((RuntimeError, AssertionError)):
            einsum_with_lora(x, weight, pattern, lora_A, lora_B,
                             scaling=1.0, dropout_layer=torch.nn.Identity())


@pytest.mark.unit
class TestMoEDimensions:
    """Tests for Mixture of Experts dimension correctness."""

    def test_moe_router_dimensions(self):
        """Test MoE router produces correct dimensions."""
        cfg = ModelConfig.gpt_small()
        cfg.use_moe = True
        cfg.num_experts = 8
        cfg.num_experts_per_tok = 2

        from pretraining.mlp.mlp import MoEMLP
        moe_mlp = MoEMLP(cfg, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, cfg.d_model)

        # Router should output [batch, seq_len, num_experts]
        router_logits = moe_mlp.router(residual)
        assert router_logits.shape == (batch_size, seq_len, cfg.num_experts)

    def test_moe_routing_probabilities_dimensions(self):
        """Test MoE routing probabilities have correct dimensions."""
        cfg = ModelConfig.gpt_small()
        cfg.use_moe = True
        cfg.num_experts = 8
        cfg.num_experts_per_tok = 2

        from pretraining.mlp.mlp import MoEMLP
        moe_mlp = MoEMLP(cfg, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, cfg.d_model)

        router_probs, top_k_probs, top_k_indices = moe_mlp._compute_routing(
            residual)

        assert router_probs.shape == (batch_size, seq_len, cfg.num_experts)
        assert top_k_probs.shape == (
            batch_size, seq_len, cfg.num_experts_per_tok)
        assert top_k_indices.shape == (
            batch_size, seq_len, cfg.num_experts_per_tok)

    def test_moe_output_dimensions(self):
        """Test MoE MLP output dimensions."""
        cfg = ModelConfig.gpt_small()
        cfg.use_moe = True
        cfg.num_experts = 8
        cfg.num_experts_per_tok = 2

        from pretraining.mlp.mlp import MoEMLP
        moe_mlp = MoEMLP(cfg, use_einops=True)
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, cfg.d_model)

        output, aux_loss = moe_mlp(residual)
        assert output.shape == residual.shape
        assert aux_loss is None or isinstance(aux_loss, torch.Tensor)
        if aux_loss is not None:
            assert aux_loss.shape == ()  # Scalar loss

    def test_moe_expert_dimensions(self):
        """Test MoE expert MLPs have correct dimensions."""
        cfg = ModelConfig.gpt_small()
        cfg.use_moe = True
        cfg.num_experts = 4
        cfg.num_experts_per_tok = 2

        from pretraining.mlp.mlp import MoEMLP
        moe_mlp = MoEMLP(cfg, use_einops=True)
        assert len(moe_mlp.experts) == cfg.num_experts

        # Each expert should be a standard MLP
        for expert in moe_mlp.experts:
            batch_size, seq_len = 2, 5
            residual = torch.randn(batch_size, seq_len, cfg.d_model)
            output = expert.expert(residual)
            assert output.shape == residual.shape


@pytest.mark.unit
class TestTrainingDimensions:
    """Tests for training-related dimensions."""

    def test_loss_dimensions(self, model_with_einops):
        """Test that loss computation produces correct dimensions."""
        model = model_with_einops
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, model.cfg.d_vocab, (batch_size, seq_len))
        targets = torch.randint(0, model.cfg.d_vocab, (batch_size, seq_len))

        logits, _ = model(tokens)
        # Loss should be scalar
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        assert loss.shape == ()
        assert isinstance(loss.item(), float)

    def test_gradient_dimensions(self, model_with_einops):
        """Test that gradients have correct dimensions matching parameters."""
        model = model_with_einops
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, model.cfg.d_vocab, (batch_size, seq_len))
        targets = torch.randint(0, model.cfg.d_vocab, (batch_size, seq_len))

        logits, _ = model(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        loss.backward()

        # Check that gradients match parameter shapes
        for name, param in model.named_parameters():
            if param.grad is not None:
                assert param.grad.shape == param.shape, \
                    f"Gradient shape mismatch for {name}: {param.grad.shape} != {param.shape}"

    def test_trainer_batch_dimensions(self, model_with_einops, training_args, device):
        """Test trainer handles batch dimensions correctly."""
        from pretraining.training.trainer import TransformerTrainer
        batch_size, seq_len = 4, 10
        X_train = torch.randint(
            0, model_with_einops.cfg.d_vocab, (100, seq_len))
        Y_train = torch.randint(
            0, model_with_einops.cfg.d_vocab, (100, seq_len))
        X_val = torch.randint(0, model_with_einops.cfg.d_vocab, (20, seq_len))
        Y_val = torch.randint(0, model_with_einops.cfg.d_vocab, (20, seq_len))

        trainer = TransformerTrainer(
            model=model_with_einops,
            args=training_args,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            device=device
        )

        # Test batch evaluation
        loss = trainer._evaluate_batch(
            X_train[:batch_size], Y_train[:batch_size])
        assert isinstance(loss, float)
        assert loss >= 0


@pytest.mark.unit
class TestFineTuningDimensions:
    """Tests for fine-tuning dimension correctness."""

    def test_sft_dataset_dimensions(self, sample_prompt_response_pairs):
        """Test SFT dataset produces correct dimensions."""
        import tempfile
        import pandas as pd
        import os
        from finetuning.data.sft_dataset import SFTDataset
        from pretraining.tokenization.tokenizer import CharacterTokenizer

        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(sample_prompt_response_pairs,
                              columns=['prompt', 'response'])
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            sample_text = " ".join(
                [p + " " + r for p, r in sample_prompt_response_pairs])
            tokenizer = CharacterTokenizer(sample_text)
            dataset = SFTDataset(csv_path, tokenizer, max_length=50)

            X_train, Y_train, masks_train = dataset.get_train_data()
            X_val, Y_val, masks_val = dataset.get_val_data()

            # Check dimensions
            assert len(X_train.shape) == 2  # [n_samples, seq_len]
            assert X_train.shape == Y_train.shape
            assert X_train.shape == masks_train.shape
            assert X_val.shape == Y_val.shape
            assert X_val.shape == masks_val.shape

            # Masks should be binary
            assert masks_train.dtype == torch.float32 or masks_train.dtype == torch.int64
            assert (masks_train == 0).logical_or(masks_train == 1).all()

        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_masked_loss_dimensions(self, model_with_einops, finetuning_args, device, sample_prompt_response_pairs):
        """Test masked loss computation has correct dimensions."""
        import tempfile
        import pandas as pd
        import os
        from finetuning.training.sft_trainer import SFTTrainer
        from finetuning.data.sft_dataset import SFTDataset
        from pretraining.tokenization.tokenizer import CharacterTokenizer

        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(sample_prompt_response_pairs,
                              columns=['prompt', 'response'])
            df.to_csv(f.name, index=False)
            csv_path = f.name

        try:
            sample_text = " ".join(
                [p + " " + r for p, r in sample_prompt_response_pairs])
            tokenizer = CharacterTokenizer(sample_text)
            dataset = SFTDataset(csv_path, tokenizer, max_length=50)
            X_train, Y_train, masks_train = dataset.get_train_data()
            X_val, Y_val, masks_val = dataset.get_val_data()

            trainer = SFTTrainer(
                model=model_with_einops,
                args=finetuning_args,
                X_train=X_train,
                Y_train=Y_train,
                masks_train=masks_train,
                X_val=X_val,
                Y_val=Y_val,
                masks_val=masks_val,
                device=device
            )

            # Test masked loss computation
            batch_size = finetuning_args.batch_size
            x_batch = X_train[:batch_size].to(device)
            y_batch = Y_train[:batch_size].to(device)
            masks_batch = masks_train[:batch_size].to(device)

            result = trainer.model(x_batch)
            logits = result[0] if isinstance(result, tuple) else result

            loss = trainer._compute_masked_loss(logits, y_batch, masks_batch)
            assert isinstance(loss, torch.Tensor)
            assert loss.shape == ()  # Scalar loss
            assert loss.item() >= 0

        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)


@pytest.mark.unit
class TestEdgeCaseDimensions:
    """Tests for edge cases and extreme dimensions."""

    def test_single_token_dimensions(self, small_config):
        """Test model handles single token sequences."""
        model = TransformerModel(small_config, use_einops=True)
        batch_size = 2
        tokens = torch.randint(0, small_config.d_vocab, (batch_size, 1))

        logits, _ = model(tokens)
        assert logits.shape == (batch_size, 1, small_config.d_vocab)

    def test_single_batch_dimension(self, small_config):
        """Test model handles single batch size."""
        model = TransformerModel(small_config, use_einops=True)
        tokens = torch.randint(0, small_config.d_vocab, (1, 5))

        logits, _ = model(tokens)
        assert logits.shape[0] == 1

    def test_large_sequence_dimensions(self, small_config):
        """Test model handles large sequences."""
        model = TransformerModel(small_config, use_einops=True)
        batch_size = 2
        seq_len = 100  # Larger than typical

        tokens = torch.randint(0, small_config.d_vocab, (batch_size, seq_len))
        logits, cache = model(tokens)

        assert logits.shape == (batch_size, seq_len, small_config.d_vocab)
        assert len(cache) == len(model.blocks)

    def test_odd_dimensions(self):
        """Test model handles odd dimension sizes."""
        base_cfg = ModelConfig.gpt_small()
        cfg_dict = {
            'd_model': 257,  # Odd
            'd_head': 65,    # Odd
            'd_mlp': 513,    # Odd
            'n_heads': 7,   # Odd
            'n_layers': 3,
            'd_vocab': 1000,
            'n_ctx': 128,
            'architecture': base_cfg.architecture,
            'positional_encoding': base_cfg.positional_encoding,
            'normalization': base_cfg.normalization,
            'activation': base_cfg.activation,
        }
        cfg = ModelConfig.from_dict(cfg_dict)

        model = TransformerModel(cfg, use_einops=True)
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, cfg.d_vocab, (batch_size, seq_len))

        logits, _ = model(tokens)
        assert logits.shape == (batch_size, seq_len, cfg.d_vocab)

    def test_very_small_dimensions(self):
        """Test model handles very small dimensions."""
        base_cfg = ModelConfig.gpt_small()
        cfg_dict = {
            'd_model': 32,
            'd_head': 8,
            'd_mlp': 64,
            'n_heads': 2,
            'n_layers': 1,
            'd_vocab': 100,
            'n_ctx': 16,
            'architecture': base_cfg.architecture,
            'positional_encoding': base_cfg.positional_encoding,
            'normalization': base_cfg.normalization,
            'activation': base_cfg.activation,
        }
        cfg = ModelConfig.from_dict(cfg_dict)

        model = TransformerModel(cfg, use_einops=True)
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, cfg.d_vocab, (batch_size, seq_len))

        logits, _ = model(tokens)
        assert logits.shape == (batch_size, seq_len, cfg.d_vocab)


@pytest.mark.unit
class TestStateDictDimensions:
    """Tests for model state dict dimension consistency."""

    def test_state_dict_dimensions_match(self, small_config):
        """Test that saved and loaded state dicts have matching dimensions."""
        model1 = TransformerModel(small_config, use_einops=True)
        model2 = TransformerModel(small_config, use_einops=True)

        # Get state dicts
        state_dict1 = model1.state_dict()
        state_dict2 = model2.state_dict()

        # Check all parameter shapes match
        assert set(state_dict1.keys()) == set(state_dict2.keys())
        for key in state_dict1.keys():
            assert state_dict1[key].shape == state_dict2[key].shape, \
                f"Shape mismatch for {key}: {state_dict1[key].shape} != {state_dict2[key].shape}"

    def test_load_state_dict_preserves_dimensions(self, small_config):
        """Test loading state dict preserves dimensions."""
        model1 = TransformerModel(small_config, use_einops=True)
        model2 = TransformerModel(small_config, use_einops=True)

        # Save and load
        state_dict = model1.state_dict()
        model2.load_state_dict(state_dict)

        # Verify dimensions are preserved
        for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
            assert param1.shape == param2.shape, \
                f"Dimension mismatch after load: {name1} {param1.shape} != {name2} {param2.shape}"


@pytest.mark.unit
class TestCacheDimensionConsistency:
    """Tests for cache dimension consistency across operations."""

    def test_cache_accumulation_dimensions(self, small_config):
        """Test cache accumulation maintains correct dimensions."""
        model = TransformerModel(small_config, use_einops=True)
        batch_size = 2

        # First forward pass
        tokens1 = torch.randint(0, small_config.d_vocab, (batch_size, 5))
        logits1, cache1 = model(tokens1, cache=None, start_pos=0)

        # Second forward pass
        tokens2 = torch.randint(0, small_config.d_vocab, (batch_size, 3))
        logits2, cache2 = model(tokens2, cache=cache1, start_pos=5)

        # Cache should accumulate: 5 + 3 = 8
        assert cache2[0][0].shape[1] == 8
        assert cache2[0][1].shape[1] == 8

        # All cache entries should have same batch size
        for k_cache, v_cache in cache2:
            assert k_cache.shape[0] == batch_size
            assert v_cache.shape[0] == batch_size

    def test_cache_dimensions_match_attention_output(self, small_config):
        """Test cache dimensions match attention output dimensions."""
        model = TransformerModel(small_config, use_einops=True)
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, small_config.d_vocab, (batch_size, seq_len))

        logits, cache = model(tokens)

        # Check cache dimensions match attention output
        for i, (k_cache, v_cache) in enumerate(cache):
            block = model.blocks[i]
            attn_output, _ = block.attn(
                torch.randn(batch_size, seq_len, small_config.d_model),
                cache=None,
                start_pos=0
            )

            # Cache should have same batch and head dimensions
            assert k_cache.shape[0] == batch_size
            assert k_cache.shape[2] == small_config.n_kv_heads
            assert v_cache.shape[0] == batch_size
            assert v_cache.shape[2] == small_config.n_kv_heads
