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


class SimpleBPETokenizer:
    """Simple Byte Pair Encoding tokenizer (educational implementation)
    
    This is a basic BPE implementation that shows how BPE works:
    1. Start with character-level vocabulary
    2. Split text into words (by whitespace)
    3. Count all pairs of consecutive tokens across all words
    4. Find the most frequent pair
    5. Merge that pair into a new token
    6. Repeat until desired vocab size
    """

    def __init__(self, text: str, vocab_size: int = 1000):
        """Train BPE tokenizer on text"""
        if not text:
            raise ValueError("Text cannot be empty for SimpleBPETokenizer")
        
        # Start with character-level vocabulary
        chars = sorted(list(set(text)))
        self.vocab = {ch: i for i, ch in enumerate(chars)}
        self.merges = []  # Store merge operations: (token1, token2) -> new_token
        
        # Split text into words and convert each word to list of characters
        # This is the standard BPE approach
        words = text.split()
        if not words:
            # If no whitespace, treat entire text as one word
            words = [text]
        tokens = [[ch for ch in word] for word in words]
        
        # Perform BPE merges until we reach desired vocab size
        num_merges = vocab_size - len(chars)
        
        for i in range(num_merges):
            # Count all pairs across all words
            pairs = {}
            for word_tokens in tokens:
                for j in range(len(word_tokens) - 1):
                    pair = (word_tokens[j], word_tokens[j + 1])
                    pairs[pair] = pairs.get(pair, 0) + 1
            
            if not pairs:
                break
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Create new token
            new_token = best_pair[0] + best_pair[1]
            new_token_id = len(self.vocab)
            self.vocab[new_token] = new_token_id
            self.merges.append(best_pair)
            
            # Merge all occurrences of the best pair in all words
            new_tokens = []
            for word_tokens in tokens:
                merged = []
                j = 0
                while j < len(word_tokens):
                    if (j < len(word_tokens) - 1 and 
                        word_tokens[j] == best_pair[0] and 
                        word_tokens[j + 1] == best_pair[1]):
                        merged.append(new_token)
                        j += 2
                    else:
                        merged.append(word_tokens[j])
                        j += 1
                new_tokens.append(merged)
            tokens = new_tokens
        
        # Create reverse mapping
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.vocab_size = len(self.vocab)
    
    def encode(self, text: str) -> List[int]:
        """Convert text to list of token IDs"""
        # Split into words and convert to character tokens
        words = text.split()
        if not words:
            words = [text]
        tokens = [[ch for ch in word] for word in words]
        
        # Apply all merges in order
        for merge_pair in self.merges:
            new_token = merge_pair[0] + merge_pair[1]
            new_tokens = []
            for word_tokens in tokens:
                merged = []
                j = 0
                while j < len(word_tokens):
                    if (j < len(word_tokens) - 1 and 
                        word_tokens[j] == merge_pair[0] and 
                        word_tokens[j + 1] == merge_pair[1]):
                        merged.append(new_token)
                        j += 2
                    else:
                        merged.append(word_tokens[j])
                        j += 1
                new_tokens.append(merged)
            tokens = new_tokens
        
        # Convert tokens to IDs and add space tokens between words
        result = []
        for i, word_tokens in enumerate(tokens):
            for token in word_tokens:
                if token in self.vocab:
                    result.append(self.vocab[token])
                else:
                    # Fallback: use character-level encoding
                    for ch in token:
                        if ch in self.vocab:
                            result.append(self.vocab[ch])
            # Add space token between words (except after last word)
            if i < len(tokens) - 1 and ' ' in self.vocab:
                result.append(self.vocab[' '])
        
        return result
    
    def decode(self, tokens: List[int]) -> str:
        """Convert list of token IDs back to text"""
        token_strings = [self.id_to_token.get(token_id, "") for token_id in tokens]
        return "".join(token_strings)
    
    def encode_tensor(self, text: str) -> torch.Tensor:
        """Convert text to tensor of token IDs"""
        return torch.tensor(self.encode(text), dtype=torch.long)
    
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

