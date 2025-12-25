from typing import Dict, List
import torch


class CharacterTokenizer:
    """Simple character-level tokenizer (Karpathy style)"""

    def __init__(self, text: str):
        # Get unique characters and create mappings
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> List[int]:
        """Convert text to list of token IDs"""
        return [self.stoi[c] for c in text]

    def decode(self, tokens: List[int]) -> str:
        """Convert list of token IDs back to text"""
        return "".join([self.itos[i] for i in tokens])

    def encode_tensor(self, text: str) -> torch.Tensor:
        """Convert text to tensor of token IDs"""
        return torch.tensor(self.encode(text), dtype=torch.long)

    def decode_tensor(self, tokens: torch.Tensor) -> str:
        """Convert tensor of token IDs back to text"""
        return self.decode(tokens.tolist())


class CharacterTokenizerWithTorch:
    """Character-level tokenizer using torch operations"""

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: Dict[int, str] = {i: ch for i, ch in enumerate(chars)}
        # Create lookup tensors for faster encoding
        self.char_to_idx = torch.zeros(256, dtype=torch.long)
        for ch, idx in self.stoi.items():
            self.char_to_idx[ord(ch)] = idx

    def encode(self, text: str) -> List[int]:
        """Convert text to list of token IDs"""
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, tokens: List[int]) -> str:
        """Convert list of token IDs back to text"""
        return "".join([self.itos.get(i, "") for i in tokens])

    def encode_tensor(self, text: str) -> torch.Tensor:
        """Convert text to tensor using torch operations"""
        # Convert string to byte array, then to tensor
        byte_array = [ord(c) for c in text]
        indices = self.char_to_idx[torch.tensor(byte_array, dtype=torch.long)]
        return indices

    def decode_tensor(self, tokens: torch.Tensor) -> str:
        """Convert tensor of token IDs back to text"""
        return self.decode(tokens.tolist())


class BPETokenizer:
    """Byte Pair Encoding tokenizer using tiktoken"""

    def __init__(self, text: str = "", model_name: str = "gpt2"):
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding(model_name)
            self.vocab_size = self.enc.n_vocab
        except ImportError as exc:
            raise ImportError(
                "tiktoken not installed. Install with: pip install tiktoken"
            ) from exc

    def encode(self, text: str) -> List[int]:
        """Convert text to list of token IDs"""
        return self.enc.encode(text)

    def decode(self, tokens: List[int]) -> str:
        """Convert list of token IDs back to text"""
        return self.enc.decode(tokens)

    def encode_tensor(self, text: str) -> torch.Tensor:
        """Convert text to tensor of token IDs"""
        return torch.tensor(self.encode(text), dtype=torch.long)

    def decode_tensor(self, tokens: torch.Tensor) -> str:
        """Convert tensor of token IDs back to text"""
        return self.decode(tokens.tolist())


class SentencePieceTokenizer:
    """SentencePiece tokenizer"""

    def __init__(self, text: str, vocab_size: int = 1000):
        try:
            import sentencepiece as spm
            import tempfile
            import os

            # Train SentencePiece model on the text
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".txt"
            ) as f:
                f.write(text)
                temp_file = f.name

            model_prefix = tempfile.mktemp()
            spm.SentencePieceTrainer.train(
                input=temp_file,
                model_prefix=model_prefix,
                vocab_size=vocab_size,
                model_type="bpe",
            )

            self.sp = spm.SentencePieceProcessor()
            self.sp.load(f"{model_prefix}.model")
            self.vocab_size = self.sp.get_piece_size()

            # Cleanup
            os.unlink(temp_file)
            os.unlink(f"{model_prefix}.model")
            os.unlink(f"{model_prefix}.vocab")
        except ImportError as exc:
            raise ImportError(
                "sentencepiece not installed. Install with: pip install sentencepiece"
            ) from exc

    def encode(self, text: str) -> List[int]:
        """Convert text to list of token IDs"""
        return self.sp.encode(text, out_type=int)

    def decode(self, tokens: List[int]) -> str:
        """Convert list of token IDs back to text"""
        return self.sp.decode(tokens)

    def encode_tensor(self, text: str) -> torch.Tensor:
        """Convert text to tensor of token IDs"""
        return torch.tensor(self.encode(text), dtype=torch.long)

    def decode_tensor(self, tokens: torch.Tensor) -> str:
        """Convert tensor of token IDs back to text"""
        return self.decode(tokens.tolist())

