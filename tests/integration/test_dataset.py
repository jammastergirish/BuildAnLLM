"""Integration tests for dataset classes."""

import pytest
import torch
from pretraining.data.dataset import TransformerDataset
from finetuning.data.sft_dataset import SFTDataset


@pytest.mark.integration
class TestTransformerDataset:
    """Tests for TransformerDataset."""

    def test_init(self, small_config, sample_text):
        """Test dataset initialization."""
        dataset = TransformerDataset(sample_text, small_config, tokenizer_type="character")
        assert dataset.cfg.d_vocab > 0
        assert len(dataset.X) > 0
        assert len(dataset.Y) > 0

    def test_create_tokenizer(self, small_config, sample_text):
        """Test tokenizer creation."""
        dataset = TransformerDataset(sample_text, small_config, tokenizer_type="character")
        assert hasattr(dataset, 'tokenizer')
        assert dataset.tokenizer.vocab_size > 0

    def test_create_sequences(self, small_config, sample_text):
        """Test sequence creation."""
        dataset = TransformerDataset(sample_text, small_config, tokenizer_type="character")
        assert dataset.X.shape[1] == small_config.n_ctx
        assert dataset.Y.shape[1] == small_config.n_ctx
        assert len(dataset.X) == len(dataset.Y)
        # Y should be X shifted by 1
        assert torch.allclose(dataset.Y[:, :-1], dataset.X[:, 1:])

    def test_split_data(self, small_config, sample_text):
        """Test data splitting."""
        dataset = TransformerDataset(sample_text, small_config, tokenizer_type="character", train_split=0.9)
        assert len(dataset.X_train) > 0
        assert len(dataset.X_val) > 0
        assert len(dataset.X_train) + len(dataset.X_val) == len(dataset.X)

    def test_get_train_data(self, small_config, sample_text):
        """Test getting training data."""
        dataset = TransformerDataset(sample_text, small_config, tokenizer_type="character")
        X_train, Y_train = dataset.get_train_data()
        assert X_train.shape == dataset.X_train.shape
        assert Y_train.shape == dataset.Y_train.shape

    def test_get_val_data(self, small_config, sample_text):
        """Test getting validation data."""
        dataset = TransformerDataset(sample_text, small_config, tokenizer_type="character")
        X_val, Y_val = dataset.get_val_data()
        assert X_val.shape == dataset.X_val.shape
        assert Y_val.shape == dataset.Y_val.shape

    def test_get_all_data(self, small_config, sample_text):
        """Test getting all data."""
        dataset = TransformerDataset(sample_text, small_config, tokenizer_type="character")
        X, Y = dataset.get_all_data()
        assert X.shape == dataset.X.shape
        assert Y.shape == dataset.Y.shape

    def test_different_tokenizers(self, small_config, sample_text):
        """Test with different tokenizer types."""
        for tokenizer_type in ["character", "bpe-simple"]:
            dataset = TransformerDataset(sample_text, small_config, tokenizer_type=tokenizer_type)
            assert dataset.tokenizer.vocab_size > 0
            assert len(dataset.X) > 0

    def test_short_text_raises(self, small_config):
        """Short texts should raise a clear error."""
        short_text = "Too short."
        with pytest.raises(ValueError, match="Text too short to create sequences"):
            TransformerDataset(short_text, small_config, tokenizer_type="character")


@pytest.mark.integration
class TestSFTDataset:
    """Tests for SFTDataset."""

    def test_init(self, sample_csv_file, character_tokenizer, small_config):
        """Test SFT dataset initialization."""
        dataset = SFTDataset(
            sample_csv_file,
            character_tokenizer,
            max_length=50,
            train_split=0.9
        )
        assert len(dataset.X) > 0
        assert len(dataset.Y) > 0
        assert len(dataset.masks) > 0

    def test_create_sequences(self, sample_csv_file, character_tokenizer):
        """Test sequence creation with masks."""
        dataset = SFTDataset(
            sample_csv_file,
            character_tokenizer,
            max_length=50
        )
        # Check shapes
        assert dataset.X.shape == dataset.Y.shape
        assert dataset.masks.shape == dataset.Y.shape
        
        # Check that masks are binary
        assert torch.all((dataset.masks == 0) | (dataset.masks == 1))

    def test_mask_prompt_tokens(self, sample_csv_file, character_tokenizer):
        """Test that prompt tokens are masked."""
        dataset = SFTDataset(
            sample_csv_file,
            character_tokenizer,
            max_length=50,
            mask_prompt=True
        )
        # Check that some tokens are masked (prompt tokens)
        assert (dataset.masks == 0).any()
        # Check that some tokens are not masked (response tokens)
        assert (dataset.masks == 1).any()

    def test_mask_prompt_false(self, temp_checkpoint_dir, character_tokenizer):
        """Test that prompt tokens are included when mask_prompt is False."""
        import pandas as pd

        prompt = "Prompt"
        response = "Response"
        csv_path = f"{temp_checkpoint_dir}/mask_prompt_false.csv"
        pd.DataFrame({"prompt": [prompt], "response": [response]}).to_csv(
            csv_path, index=False
        )
        max_length = len(character_tokenizer.encode(prompt)) + len(
            character_tokenizer.encode(response)
        )
        dataset = SFTDataset(
            csv_path,
            character_tokenizer,
            max_length=max_length,
            mask_prompt=False,
        )
        assert torch.all(dataset.masks == 1)

    def test_empty_responses_skipped(self, temp_checkpoint_dir, character_tokenizer):
        """Test that empty responses are skipped."""
        import pandas as pd

        csv_path = f"{temp_checkpoint_dir}/skip_empty.csv"
        df = pd.DataFrame({
            "prompt": ["Q1", "Q2"],
            "response": ["", "A2"]
        })
        df.to_csv(csv_path, index=False)

        dataset = SFTDataset(csv_path, character_tokenizer, max_length=50)
        assert len(dataset.prompts) == 1
        assert dataset.X.shape[0] == 1
        assert dataset.masks.sum().item() > 0

    def test_split_data(self, sample_csv_file, character_tokenizer):
        """Test data splitting."""
        dataset = SFTDataset(
            sample_csv_file,
            character_tokenizer,
            max_length=50,
            train_split=0.9
        )
        assert len(dataset.X_train) > 0
        assert len(dataset.X_val) > 0
        assert len(dataset.masks_train) > 0
        assert len(dataset.masks_val) > 0

    def test_get_train_data(self, sample_csv_file, character_tokenizer):
        """Test getting training data."""
        dataset = SFTDataset(
            sample_csv_file,
            character_tokenizer,
            max_length=50
        )
        X_train, Y_train, masks_train = dataset.get_train_data()
        assert X_train.shape == dataset.X_train.shape
        assert Y_train.shape == dataset.Y_train.shape
        assert masks_train.shape == dataset.masks_train.shape

    def test_get_val_data(self, sample_csv_file, character_tokenizer):
        """Test getting validation data."""
        dataset = SFTDataset(
            sample_csv_file,
            character_tokenizer,
            max_length=50
        )
        X_val, Y_val, masks_val = dataset.get_val_data()
        assert X_val.shape == dataset.X_val.shape
        assert Y_val.shape == dataset.Y_val.shape
        assert masks_val.shape == dataset.masks_val.shape

    def test_empty_responses_error(self, temp_checkpoint_dir, character_tokenizer):
        """Test that empty responses raise error."""
        import pandas as pd
        csv_path = f"{temp_checkpoint_dir}/empty_responses.csv"
        df = pd.DataFrame({
            'prompt': ['Q1', 'Q2'],
            'response': ['', '']  # All empty
        })
        df.to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError, match="All responses are empty"):
            SFTDataset(csv_path, character_tokenizer, max_length=50)

    def test_instruction_output_format(self, temp_checkpoint_dir, character_tokenizer):
        """Test instruction/output column format."""
        import pandas as pd
        csv_path = f"{temp_checkpoint_dir}/instruction_format.csv"
        df = pd.DataFrame({
            'instruction': ['Q1', 'Q2'],
            'output': ['A1', 'A2']
        })
        df.to_csv(csv_path, index=False)
        
        dataset = SFTDataset(csv_path, character_tokenizer, max_length=50)
        assert len(dataset.X) > 0
