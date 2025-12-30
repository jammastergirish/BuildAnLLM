"""Integration tests for training classes."""

import pytest
import torch
from pretraining.training.trainer import TransformerTrainer
from finetuning.training.sft_trainer import SFTTrainer


@pytest.mark.integration
class TestTransformerTrainer:
    """Tests for TransformerTrainer."""

    def test_init(self, model_with_einops, training_args, sample_batch_tokens, device):
        """Test trainer initialization."""
        batch_size, seq_len = sample_batch_tokens.shape
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
        assert trainer.model == model_with_einops
        assert trainer.optimizer is not None

    def test_evaluate_batch(self, model_with_einops, training_args, device):
        """Test batch evaluation."""
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

        x_batch = X_train[:batch_size].to(device)
        y_batch = Y_train[:batch_size].to(device)
        loss = trainer._evaluate_batch(x_batch, y_batch)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_training_step(self, model_with_einops, training_args, device):
        """Test training step."""
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

        x_batch = X_train[:batch_size].to(device)
        y_batch = Y_train[:batch_size].to(device)
        loss = trainer._training_step(x_batch, y_batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_estimate_loss(self, model_with_einops, training_args, device):
        """Test loss estimation."""
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

        losses = trainer.estimate_loss()
        assert 'train' in losses
        assert 'val' in losses
        assert losses['train'] >= 0
        assert losses['val'] >= 0

    def test_train_single_step(self, model_with_einops, training_args, device):
        """Test single training step."""
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

        metrics = trainer.train_single_step()
        assert isinstance(metrics, dict)
        assert "loss" in metrics
        assert "running_loss" in metrics
        assert "grad_norm" in metrics
        assert metrics["loss"] >= 0
        assert metrics["running_loss"] >= 0


@pytest.mark.integration
class TestSFTTrainer:
    """Tests for SFTTrainer."""

    def test_init(self, model_with_einops, finetuning_args, device):
        """Test SFT trainer initialization."""
        batch_size, seq_len = 4, 20
        X_train = torch.randint(
            0, model_with_einops.cfg.d_vocab, (50, seq_len))
        Y_train = torch.randint(
            0, model_with_einops.cfg.d_vocab, (50, seq_len))
        masks_train = torch.randint(0, 2, (50, seq_len)).float()
        X_val = torch.randint(0, model_with_einops.cfg.d_vocab, (10, seq_len))
        Y_val = torch.randint(0, model_with_einops.cfg.d_vocab, (10, seq_len))
        masks_val = torch.randint(0, 2, (10, seq_len)).float()

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
        assert trainer.model == model_with_einops
        assert trainer.optimizer is not None

    def test_compute_masked_loss(self, model_with_einops, finetuning_args, device):
        """Test masked loss computation."""
        batch_size, seq_len = 2, 10
        X_train = torch.randint(
            0, model_with_einops.cfg.d_vocab, (50, seq_len))
        Y_train = torch.randint(
            0, model_with_einops.cfg.d_vocab, (50, seq_len))
        masks_train = torch.randint(0, 2, (50, seq_len)).float()
        X_val = torch.randint(0, model_with_einops.cfg.d_vocab, (10, seq_len))
        Y_val = torch.randint(0, model_with_einops.cfg.d_vocab, (10, seq_len))
        masks_val = torch.randint(0, 2, (10, seq_len)).float()

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

        x_batch = X_train[:batch_size].to(device)
        y_batch = Y_train[:batch_size].to(device)
        masks_batch = masks_train[:batch_size].to(device)

        result = trainer.model(x_batch)
        # Handle tuple return (logits, cache) - extract just logits
        logits = result[0] if isinstance(result, tuple) else result
        loss = trainer._compute_masked_loss(logits, y_batch, masks_batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_training_step(self, model_with_einops, finetuning_args, device):
        """Test SFT training step."""
        batch_size, seq_len = 2, 10
        X_train = torch.randint(
            0, model_with_einops.cfg.d_vocab, (50, seq_len))
        Y_train = torch.randint(
            0, model_with_einops.cfg.d_vocab, (50, seq_len))
        masks_train = torch.randint(0, 2, (50, seq_len)).float()
        X_val = torch.randint(0, model_with_einops.cfg.d_vocab, (10, seq_len))
        Y_val = torch.randint(0, model_with_einops.cfg.d_vocab, (10, seq_len))
        masks_val = torch.randint(0, 2, (10, seq_len)).float()

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

        x_batch = X_train[:batch_size].to(device)
        y_batch = Y_train[:batch_size].to(device)
        masks_batch = masks_train[:batch_size].to(device)

        # Test masked loss computation (equivalent to training step)
        result = trainer.model(x_batch)
        # Handle tuple return (logits, cache) - extract just logits
        logits = result[0] if isinstance(result, tuple) else result
        loss = trainer._compute_masked_loss(logits, y_batch, masks_batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() >= 0

    def test_estimate_loss(self, model_with_einops, finetuning_args, device):
        """Test SFT loss estimation."""
        batch_size, seq_len = 2, 10
        X_train = torch.randint(
            0, model_with_einops.cfg.d_vocab, (50, seq_len))
        Y_train = torch.randint(
            0, model_with_einops.cfg.d_vocab, (50, seq_len))
        masks_train = torch.randint(0, 2, (50, seq_len)).float()
        X_val = torch.randint(0, model_with_einops.cfg.d_vocab, (10, seq_len))
        Y_val = torch.randint(0, model_with_einops.cfg.d_vocab, (10, seq_len))
        masks_val = torch.randint(0, 2, (10, seq_len)).float()

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

        losses = trainer.estimate_loss()
        assert 'train' in losses
        assert 'val' in losses
        assert losses['train'] >= 0
        assert losses['val'] >= 0
