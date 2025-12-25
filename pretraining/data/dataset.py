import torch
from typing import Tuple
from config import ModelConfig
from pretraining.tokenization.tokenizer import (
    CharacterTokenizer,
    CharacterTokenizerWithTorch,
    BPETokenizer,
    SentencePieceTokenizer,
)


class TransformerDataset:
    def __init__(
        self,
        text: str,
        cfg: ModelConfig,
        tokenizer_type: str = "character",
        train_split: float = 0.9,
    ):
        self.text = text
        self.cfg = cfg
        self.tokenizer_type = tokenizer_type
        self.train_split = train_split

        # Create tokenizer
        self._create_tokenizer()

        # Update config to match tokenizer vocab size
        self.cfg.d_vocab = self.tokenizer.vocab_size

        # Encode text to tokens
        # self.data: [total_tokens] - flattened tensor of token IDs
        self.data = self.tokenizer.encode_tensor(text)

        # Create dataset
        self.X, self.Y = self._create_sequences()

        # Split into train/val
        self.X_train, self.Y_train, self.X_val, self.Y_val = self._split_data()

    def _create_tokenizer(self):
        """Create tokenizer based on type"""
        if self.tokenizer_type == "character":
            self.tokenizer = CharacterTokenizer(self.text)
        elif self.tokenizer_type == "character_torch":
            self.tokenizer = CharacterTokenizerWithTorch(self.text)
        elif self.tokenizer_type == "bpe":
            self.tokenizer = BPETokenizer(self.text)
        elif self.tokenizer_type == "sentencepiece":
            self.tokenizer = SentencePieceTokenizer(
                self.text, vocab_size=min(self.cfg.d_vocab, 10000)
            )
        else:
            raise ValueError(f"Unknown tokenizer type: {self.tokenizer_type}")

        print(f"Using {self.tokenizer_type} tokenizer")
        print(f"Vocabulary size: {self.tokenizer.vocab_size}")

    def _create_sequences(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create (input, target) pairs where target is input shifted by 1"""
        block_size = self.cfg.n_ctx
        # self.data: [total_tokens] - flattened token sequence
        X = []
        Y = []
        for i in range(len(self.data) - block_size):
            # X[i]: [block_size] - input sequence
            # Y[i]: [block_size] - target sequence (shifted by 1)
            X.append(self.data[i: i + block_size])
            Y.append(self.data[i + 1: i + block_size + 1])
        # X: [num_sequences, block_size]
        # Y: [num_sequences, block_size]
        return torch.stack(X), torch.stack(Y)

    def _split_data(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split data into train and validation sets"""
        # self.X: [num_sequences, block_size]
        # self.Y: [num_sequences, block_size]
        split_idx = int(self.train_split * len(self.X))
        # X_train: [num_train_sequences, block_size]
        # Y_train: [num_train_sequences, block_size]
        X_train = self.X[:split_idx]
        Y_train = self.Y[:split_idx]
        # X_val: [num_val_sequences, block_size]
        # Y_val: [num_val_sequences, block_size]
        X_val = self.X[split_idx:]
        Y_val = self.Y[split_idx:]
        return X_train, Y_train, X_val, Y_val

    def get_train_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get training data"""
        return self.X_train, self.Y_train

    def get_val_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get validation data"""
        return self.X_val, self.Y_val

    def get_all_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get all data (train + val)"""
        return self.X, self.Y

    def print_info(self):
        """Print dataset information"""
        print(f"Total tokens: {len(self.data)}")
        print(f"Block size (sequence length): {self.cfg.n_ctx}")
        print(f"Created {len(self.X)} training sequences")
        print(f"Input shape: {self.X.shape}, Target shape: {self.Y.shape}")
        print(
            f"Train: {len(self.X_train)} sequences, Val: {len(self.X_val)} sequences"
        )

