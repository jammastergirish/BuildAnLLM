"""Integration tests for LoRA fine-tuning workflows."""

import pytest
import torch
from config import ModelConfig
from pretraining.model.model import TransformerModel
from finetuning.peft.lora_utils import convert_model_to_lora
from finetuning.training.sft_trainer import SFTTrainer
from finetuning.training.finetuning_args import FinetuningArgs
from finetuning.data.sft_dataset import SFTDataset
from pretraining.tokenization.tokenizer import CharacterTokenizer


@pytest.mark.integration
class TestLoraDevicePlacement:
    """Tests for LoRA device placement and consistency."""

    def test_lora_matrices_on_correct_device(self, model_with_einops, device):
        """Test that LoRA matrices are created on the correct device."""
        model = model_with_einops.to(device)
        model_lora = convert_model_to_lora(model, rank=8, alpha=8.0, dropout=0.0)
        
        # Check that all LoRA matrices are on the correct device
        for name, param in model_lora.named_parameters():
            if 'lora' in name:
                assert param.device == device, f"LoRA parameter {name} is on {param.device}, expected {device}"

    def test_lora_forward_on_device(self, model_with_einops, device):
        """Test that LoRA forward pass works on the specified device."""
        model = model_with_einops.to(device)
        model_lora = convert_model_to_lora(model, rank=8, alpha=8.0, dropout=0.0)
        
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, model.cfg.d_vocab, (batch_size, seq_len)).to(device)
        
        result = model_lora(tokens)
        logits = result[0] if isinstance(result, tuple) else result
        
        assert logits.device == device
        assert logits.shape == (batch_size, seq_len, model.cfg.d_vocab)

    def test_lora_model_move_to_device(self, model_with_einops):
        """Test that LoRA matrices move correctly when model is moved."""
        model_lora = convert_model_to_lora(model_with_einops, rank=8, alpha=8.0, dropout=0.0)
        
        # Initially on CPU
        assert next(model_lora.parameters()).device.type == 'cpu'
        
        # Move to MPS if available, else CUDA if available, else CPU
        if torch.backends.mps.is_available():
            target_device = torch.device('mps')
        elif torch.cuda.is_available():
            target_device = torch.device('cuda')
        else:
            target_device = torch.device('cpu')
        
        model_lora = model_lora.to(target_device)
        
        # Check that all parameters (including LoRA) moved
        for name, param in model_lora.named_parameters():
            assert param.device.type == target_device.type, \
                f"Parameter {name} is on {param.device}, expected {target_device}"


@pytest.mark.integration
class TestLoraTrainingWorkflow:
    """Tests for LoRA fine-tuning training workflows."""

    def test_lora_training_step(self, model_with_einops, finetuning_args, device, sample_prompt_response_pairs):
        """Test LoRA training step with proper logits extraction."""
        import tempfile
        import pandas as pd
        import os
        
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(sample_prompt_response_pairs, columns=['prompt', 'response'])
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            # Create tokenizer
            sample_text = " ".join([p + " " + r for p, r in sample_prompt_response_pairs])
            tokenizer = CharacterTokenizer(sample_text)
            
            # Create dataset
            dataset = SFTDataset(csv_path, tokenizer, max_length=50)
            X_train, Y_train, masks_train = dataset.get_train_data()
            X_val, Y_val, masks_val = dataset.get_val_data()
            
            # Convert model to LoRA
            model_lora = convert_model_to_lora(
                model_with_einops,
                rank=8,
                alpha=8.0,
                dropout=0.0
            )
            model_lora = model_lora.to(device)
            
            # Create trainer
            trainer = SFTTrainer(
                model=model_lora,
                args=finetuning_args,
                X_train=X_train,
                Y_train=Y_train,
                masks_train=masks_train,
                X_val=X_val,
                Y_val=Y_val,
                masks_val=masks_val,
                device=device
            )
            
            # Test training step
            batch_size = finetuning_args.batch_size
            x_batch = X_train[:batch_size].to(device)
            y_batch = Y_train[:batch_size].to(device)
            masks_batch = masks_train[:batch_size].to(device)
            
            # Forward pass should return tuple (logits, cache)
            result = trainer.model(x_batch)
            assert isinstance(result, tuple), "Model should return tuple (logits, cache)"
            logits, cache = result
            assert logits.device == device
            assert logits.shape[0] == batch_size
            
            # Compute loss
            loss = trainer._compute_masked_loss(logits, y_batch, masks_batch)
            assert isinstance(loss, torch.Tensor)
            assert loss.device == device
            assert loss.item() >= 0
            
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_lora_evaluate_batch(self, model_with_einops, finetuning_args, device, sample_prompt_response_pairs):
        """Test LoRA evaluation batch with proper logits extraction."""
        import tempfile
        import pandas as pd
        import os
        
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(sample_prompt_response_pairs, columns=['prompt', 'response'])
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            # Create tokenizer
            sample_text = " ".join([p + " " + r for p, r in sample_prompt_response_pairs])
            tokenizer = CharacterTokenizer(sample_text)
            
            # Create dataset
            dataset = SFTDataset(csv_path, tokenizer, max_length=50)
            X_train, Y_train, masks_train = dataset.get_train_data()
            X_val, Y_val, masks_val = dataset.get_val_data()
            
            # Convert model to LoRA
            model_lora = convert_model_to_lora(
                model_with_einops,
                rank=8,
                alpha=8.0,
                dropout=0.0
            )
            model_lora = model_lora.to(device)
            
            # Create trainer
            trainer = SFTTrainer(
                model=model_lora,
                args=finetuning_args,
                X_train=X_train,
                Y_train=Y_train,
                masks_train=masks_train,
                X_val=X_val,
                Y_val=Y_val,
                masks_val=masks_val,
                device=device
            )
            
            # Test evaluation batch
            batch_size = finetuning_args.batch_size
            x_batch = X_train[:batch_size].to(device)
            y_batch = Y_train[:batch_size].to(device)
            masks_batch = masks_train[:batch_size].to(device)
            
            # This should work without errors
            loss = trainer._evaluate_batch(x_batch, y_batch, masks_batch)
            assert isinstance(loss, float)
            assert loss >= 0
            
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_lora_training_ui_step(self, model_with_einops, finetuning_args, device, sample_prompt_response_pairs):
        """Test LoRA training UI step function."""
        import tempfile
        import pandas as pd
        import os
        from finetuning.training.sft_trainer import SFTTrainer
        
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(sample_prompt_response_pairs, columns=['prompt', 'response'])
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            # Create tokenizer
            sample_text = " ".join([p + " " + r for p, r in sample_prompt_response_pairs])
            tokenizer = CharacterTokenizer(sample_text)
            
            # Create dataset
            dataset = SFTDataset(csv_path, tokenizer, max_length=50)
            X_train, Y_train, masks_train = dataset.get_train_data()
            X_val, Y_val, masks_val = dataset.get_val_data()
            
            # Convert model to LoRA
            model_lora = convert_model_to_lora(
                model_with_einops,
                rank=8,
                alpha=8.0,
                dropout=0.0
            )
            model_lora = model_lora.to(device)
            
            # Create trainer
            trainer = SFTTrainer(
                model=model_lora,
                args=finetuning_args,
                X_train=X_train,
                Y_train=Y_train,
                masks_train=masks_train,
                X_val=X_val,
                Y_val=Y_val,
                masks_val=masks_val,
                device=device
            )
            
            # Test training step (this is what the UI uses)
            metrics = trainer.train_single_step()
            loss = metrics["loss"]
            running_loss = metrics["running_loss"]
            
            assert isinstance(loss, float)
            assert isinstance(running_loss, float)
            assert loss >= 0
            assert running_loss >= 0
            
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)


@pytest.mark.integration
class TestModelReturnFormat:
    """Tests to ensure model return format is handled correctly everywhere."""

    def test_model_always_returns_tuple_with_cache(self, model_with_einops, device):
        """Test that model always returns tuple (logits, cache) when blocks exist."""
        model = model_with_einops.to(device)
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, model.cfg.d_vocab, (batch_size, seq_len)).to(device)
        
        result = model(tokens)
        assert isinstance(result, tuple), "Model should return tuple (logits, cache)"
        assert len(result) == 2
        logits, cache = result
        assert logits.device == device
        assert isinstance(cache, list)
        assert len(cache) == len(model.blocks)

    def test_sft_trainer_handles_tuple_return(self, model_with_einops, finetuning_args, device, sample_prompt_response_pairs):
        """Test that SFTTrainer handles tuple return correctly."""
        import tempfile
        import pandas as pd
        import os
        
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df = pd.DataFrame(sample_prompt_response_pairs, columns=['prompt', 'response'])
            df.to_csv(f.name, index=False)
            csv_path = f.name
        
        try:
            # Create tokenizer
            sample_text = " ".join([p + " " + r for p, r in sample_prompt_response_pairs])
            tokenizer = CharacterTokenizer(sample_text)
            
            # Create dataset
            dataset = SFTDataset(csv_path, tokenizer, max_length=50)
            X_train, Y_train, masks_train = dataset.get_train_data()
            X_val, Y_val, masks_val = dataset.get_val_data()
            
            model = model_with_einops.to(device)
            trainer = SFTTrainer(
                model=model,
                args=finetuning_args,
                X_train=X_train,
                Y_train=Y_train,
                masks_train=masks_train,
                X_val=X_val,
                Y_val=Y_val,
                masks_val=masks_val,
                device=device
            )
            
            # Test that _evaluate_batch handles tuple return
            batch_size = finetuning_args.batch_size
            x_batch = X_train[:batch_size].to(device)
            y_batch = Y_train[:batch_size].to(device)
            masks_batch = masks_train[:batch_size].to(device)
            
            loss = trainer._evaluate_batch(x_batch, y_batch, masks_batch)
            assert isinstance(loss, float)
            assert loss >= 0
            
        finally:
            if os.path.exists(csv_path):
                os.unlink(csv_path)

    def test_lora_attention_returns_tuple(self, model_with_einops, device):
        """Test that LoRA attention returns tuple (output, cache)."""
        model_lora = convert_model_to_lora(model_with_einops, rank=8, alpha=8.0, dropout=0.0)
        model_lora = model_lora.to(device)
        
        attn = model_lora.blocks[0].attn
        batch_size, seq_len = 2, 5
        residual = torch.randn(batch_size, seq_len, model_with_einops.cfg.d_model).to(device)
        
        result = attn(residual, cache=None, start_pos=0)
        assert isinstance(result, tuple)
        assert len(result) == 2
        output, cache = result
        assert output.device == device
        assert isinstance(cache, tuple)
        assert len(cache) == 2


@pytest.mark.integration
class TestDeviceConsistency:
    """Tests to ensure all tensors are on the correct device."""

    def test_all_model_parameters_on_device(self, model_with_einops, device):
        """Test that all model parameters are on the correct device."""
        model = model_with_einops.to(device)
        
        for name, param in model.named_parameters():
            assert param.device == device, f"Parameter {name} is on {param.device}, expected {device}"

    def test_all_lora_parameters_on_device(self, model_with_einops, device):
        """Test that all LoRA parameters are on the correct device."""
        model_lora = convert_model_to_lora(model_with_einops, rank=8, alpha=8.0, dropout=0.0)
        model_lora = model_lora.to(device)
        
        for name, param in model_lora.named_parameters():
            assert param.device == device, f"Parameter {name} is on {param.device}, expected {device}"

    def test_lora_forward_outputs_on_device(self, model_with_einops, device):
        """Test that LoRA forward pass outputs are on the correct device."""
        model_lora = convert_model_to_lora(model_with_einops, rank=8, alpha=8.0, dropout=0.0)
        model_lora = model_lora.to(device)
        
        batch_size, seq_len = 2, 5
        tokens = torch.randint(0, model_with_einops.cfg.d_vocab, (batch_size, seq_len)).to(device)
        
        result = model_lora(tokens)
        logits, cache = result
        
        assert logits.device == device
        assert isinstance(cache, list)
        for k_cache, v_cache in cache:
            assert k_cache.device == device
            assert v_cache.device == device

    def test_lora_einsum_device_match(self, model_with_einops, device):
        """Test that LoRA einsum operations handle device correctly."""
        from finetuning.peft.lora_wrappers import einsum_with_lora
        import einops
        
        model_lora = convert_model_to_lora(model_with_einops, rank=8, alpha=8.0, dropout=0.0)
        model_lora = model_lora.to(device)
        
        attn = model_lora.blocks[0].attn
        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, model_with_einops.cfg.d_model).to(device)
        
        # Test einsum_with_lora with device mismatch handling
        weight = attn.W_Q.to(device)
        lora_A = attn.W_Q_lora_A.to(device)
        lora_B = attn.W_Q_lora_B.to(device)
        scaling = attn.W_Q_lora_scaling
        dropout_layer = attn.W_Q_lora_dropout
        
        pattern = "batch posn d_model, n_heads d_head d_model -> batch posn n_heads d_head"
        
        # This should work without device errors
        output = einsum_with_lora(x, weight, pattern, lora_A, lora_B, scaling, dropout_layer)
        assert output.device == device
        assert output.shape == (batch_size, seq_len, model_with_einops.cfg.n_heads, model_with_einops.cfg.d_head)

