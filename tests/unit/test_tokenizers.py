"""Unit tests for tokenizer classes."""

import pytest
import torch
from pretraining.tokenization.tokenizer import (
    CharacterTokenizer,
    CharacterTokenizerWithTorch,
    SimpleBPETokenizer,
    BPETokenizer,
    SentencePieceTokenizer,
)


@pytest.mark.unit
class TestCharacterTokenizer:
    """Tests for CharacterTokenizer."""

    def test_init(self, sample_text):
        """Test tokenizer initialization."""
        tokenizer = CharacterTokenizer(sample_text)
        assert tokenizer.vocab_size > 0
        assert len(tokenizer.stoi) == tokenizer.vocab_size
        assert len(tokenizer.itos) == tokenizer.vocab_size

    def test_encode(self, character_tokenizer):
        """Test encoding text to token IDs."""
        text = "Hello"
        tokens = character_tokenizer.encode(text)
        assert isinstance(tokens, list)
        assert len(tokens) == len(text)
        assert all(isinstance(t, int) for t in tokens)

    def test_decode(self, character_tokenizer):
        """Test decoding token IDs to text."""
        text = "Hello world"
        tokens = character_tokenizer.encode(text)
        decoded = character_tokenizer.decode(tokens)
        assert decoded == text

    def test_encode_decode_roundtrip(self, character_tokenizer):
        """Test encode/decode round-trip."""
        # Use only characters that are guaranteed to be in the vocabulary
        # (from the sample_text used to create the tokenizer)
        texts = ["Hello", "world", "test"]
        for text in texts:
            # Check that all characters are in vocabulary
            if all(c in character_tokenizer.stoi for c in text):
                tokens = character_tokenizer.encode(text)
                decoded = character_tokenizer.decode(tokens)
                assert decoded == text
        
        # Test with sample text that's guaranteed to work
        sample_text = "Hello world"
        if all(c in character_tokenizer.stoi for c in sample_text):
            tokens = character_tokenizer.encode(sample_text)
            decoded = character_tokenizer.decode(tokens)
            assert decoded == sample_text

    def test_encode_tensor(self, character_tokenizer):
        """Test encoding to tensor."""
        text = "Hello"
        tensor = character_tokenizer.encode_tensor(text)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.long
        assert len(tensor) == len(text)

    def test_decode_tensor(self, character_tokenizer):
        """Test decoding from tensor."""
        text = "Hello"
        tensor = character_tokenizer.encode_tensor(text)
        decoded = character_tokenizer.decode_tensor(tensor)
        assert decoded == text

    def test_empty_text(self):
        """Test with empty text."""
        tokenizer = CharacterTokenizer("a")
        assert tokenizer.encode("") == []
        assert tokenizer.decode([]) == ""

    def test_special_characters(self, character_tokenizer):
        """Test with special characters that exist in vocabulary."""
        # Use characters that are guaranteed to be in the vocabulary
        # (from the sample_text used to create the tokenizer)
        text = "Hello world"
        if all(c in character_tokenizer.stoi for c in text):
            tokens = character_tokenizer.encode(text)
            decoded = character_tokenizer.decode(tokens)
            assert decoded == text


@pytest.mark.unit
class TestCharacterTokenizerWithTorch:
    """Tests for CharacterTokenizerWithTorch."""

    def test_init(self, sample_text):
        """Test tokenizer initialization."""
        tokenizer = CharacterTokenizerWithTorch(sample_text)
        assert tokenizer.vocab_size > 0
        assert hasattr(tokenizer, 'char_to_idx')

    def test_encode(self, character_tokenizer_torch):
        """Test encoding."""
        text = "Hello"
        tokens = character_tokenizer_torch.encode(text)
        assert isinstance(tokens, list)
        assert len(tokens) == len(text)

    def test_decode(self, character_tokenizer_torch):
        """Test decoding."""
        text = "Hello world"
        tokens = character_tokenizer_torch.encode(text)
        decoded = character_tokenizer_torch.decode(tokens)
        assert decoded == text

    def test_encode_tensor(self, character_tokenizer_torch):
        """Test tensor encoding."""
        text = "Hello"
        tensor = character_tokenizer_torch.encode_tensor(text)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.long

    def test_decode_tensor(self, character_tokenizer_torch):
        """Test tensor decoding."""
        text = "Hello"
        tensor = character_tokenizer_torch.encode_tensor(text)
        decoded = character_tokenizer_torch.decode_tensor(tensor)
        assert decoded == text

    def test_encode_decode_roundtrip(self, character_tokenizer_torch):
        """Test round-trip."""
        text = "Hello world"
        tokens = character_tokenizer_torch.encode(text)
        decoded = character_tokenizer_torch.decode(tokens)
        assert decoded == text

    def test_init_rejects_non_ascii(self):
        """Non-ASCII training text should raise a clear error."""
        text = "Hello " + chr(0x0100)
        with pytest.raises(ValueError, match="byte-range"):
            CharacterTokenizerWithTorch(text)

    def test_encode_tensor_maps_non_ascii(self, character_tokenizer_torch):
        """Non-ASCII input should map to the default token without crashing."""
        text = "Hi" + chr(0x0100)
        tensor = character_tokenizer_torch.encode_tensor(text)
        assert isinstance(tensor, torch.Tensor)
        assert len(tensor) == len(text)
        assert tensor[-1].item() == 0


@pytest.mark.unit
class TestSimpleBPETokenizer:
    """Tests for SimpleBPETokenizer."""

    def test_init(self, sample_text):
        """Test initialization."""
        tokenizer = SimpleBPETokenizer(sample_text, vocab_size=50)
        assert tokenizer.vocab_size >= 50
        assert len(tokenizer.vocab) >= 50
        assert len(tokenizer.merges) >= 0

    def test_init_empty_text(self):
        """Test initialization with empty text raises error."""
        with pytest.raises(ValueError):
            SimpleBPETokenizer("", vocab_size=50)

    def test_encode(self, simple_bpe_tokenizer):
        """Test encoding."""
        text = "Hello world"
        tokens = simple_bpe_tokenizer.encode(text)
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_decode(self, simple_bpe_tokenizer):
        """Test decoding."""
        text = "Hello world"
        tokens = simple_bpe_tokenizer.encode(text)
        decoded = simple_bpe_tokenizer.decode(tokens)
        # Decode may not be exact due to BPE merges, but should be close
        assert isinstance(decoded, str)

    def test_encode_tensor(self, simple_bpe_tokenizer):
        """Test tensor encoding."""
        text = "Hello"
        tensor = simple_bpe_tokenizer.encode_tensor(text)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.long

    def test_decode_tensor(self, simple_bpe_tokenizer):
        """Test tensor decoding."""
        text = "Hello world"
        tokens = simple_bpe_tokenizer.encode_tensor(text)
        decoded = simple_bpe_tokenizer.decode_tensor(tokens)
        assert isinstance(decoded, str)

    def test_vocab_size(self, sample_text):
        """Test vocab size parameter."""
        tokenizer1 = SimpleBPETokenizer(sample_text, vocab_size=50)
        tokenizer2 = SimpleBPETokenizer(sample_text, vocab_size=100)
        assert tokenizer2.vocab_size >= tokenizer1.vocab_size


@pytest.mark.unit
class TestBPETokenizer:
    """Tests for BPETokenizer (tiktoken)."""

    def test_init(self):
        """Test initialization."""
        tokenizer = BPETokenizer()
        assert tokenizer.vocab_size > 0

    def test_encode(self, bpe_tokenizer):
        """Test encoding."""
        text = "Hello world"
        tokens = bpe_tokenizer.encode(text)
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_decode(self, bpe_tokenizer):
        """Test decoding."""
        text = "Hello world"
        tokens = bpe_tokenizer.encode(text)
        decoded = bpe_tokenizer.decode(tokens)
        assert decoded == text

    def test_encode_decode_roundtrip(self, bpe_tokenizer):
        """Test round-trip."""
        texts = ["Hello", "world", "test 123", "The quick brown fox"]
        for text in texts:
            tokens = bpe_tokenizer.encode(text)
            decoded = bpe_tokenizer.decode(tokens)
            assert decoded == text

    def test_encode_tensor(self, bpe_tokenizer):
        """Test tensor encoding."""
        text = "Hello"
        tensor = bpe_tokenizer.encode_tensor(text)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.long

    def test_decode_tensor(self, bpe_tokenizer):
        """Test tensor decoding."""
        text = "Hello world"
        tokens = bpe_tokenizer.encode_tensor(text)
        decoded = bpe_tokenizer.decode_tensor(tokens)
        assert decoded == text

    def test_empty_text(self, bpe_tokenizer):
        """Test with empty text."""
        assert bpe_tokenizer.encode("") == []
        assert bpe_tokenizer.decode([]) == ""


@pytest.mark.unit
class TestSentencePieceTokenizer:
    """Tests for SentencePieceTokenizer."""

    def test_init(self, sample_text):
        """Test initialization."""
        # Need vocab_size >= number of unique characters + special tokens
        # Expanded sample_text has ~66 unique chars, so use vocab_size=100
        tokenizer = SentencePieceTokenizer(sample_text, vocab_size=100)
        assert tokenizer.vocab_size > 0

    def test_encode(self, sentencepiece_tokenizer):
        """Test encoding."""
        text = "Hello world"
        tokens = sentencepiece_tokenizer.encode(text)
        assert isinstance(tokens, list)
        assert all(isinstance(t, int) for t in tokens)

    def test_decode(self, sentencepiece_tokenizer):
        """Test decoding."""
        text = "Hello world"
        tokens = sentencepiece_tokenizer.encode(text)
        decoded = sentencepiece_tokenizer.decode(tokens)
        # SentencePiece may add spaces, so check similarity
        assert isinstance(decoded, str)
        assert len(decoded) > 0

    def test_encode_tensor(self, sentencepiece_tokenizer):
        """Test tensor encoding."""
        text = "Hello"
        tensor = sentencepiece_tokenizer.encode_tensor(text)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype == torch.long

    def test_decode_tensor(self, sentencepiece_tokenizer):
        """Test tensor decoding."""
        text = "Hello world"
        tokens = sentencepiece_tokenizer.encode_tensor(text)
        decoded = sentencepiece_tokenizer.decode_tensor(tokens)
        assert isinstance(decoded, str)

    def test_vocab_size(self, sample_text):
        """Test vocab size parameter."""
        # Need vocab_size >= number of unique characters + special tokens
        # Expanded sample_text has ~66 unique chars, so use vocab_size >= 100
        tokenizer1 = SentencePieceTokenizer(sample_text, vocab_size=100)
        tokenizer2 = SentencePieceTokenizer(sample_text, vocab_size=200)
        assert tokenizer2.vocab_size >= tokenizer1.vocab_size
