# /// script
# dependencies = ["torch", "einops", "jaxtyping", "numpy", "tiktoken", "sentencepiece"]
# ///

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from jaxtyping import Float
from torch import Tensor
from config import GPTConfig
from layernorm import LayerNormWithEinops, LayerNormWithoutEinops, LayerNormWithTorch
from embed import EmbedWithoutTorch, EmbedWithTorch
from positional_embedding import PosEmbedWithEinops, PosEmbedWithoutEinops
from attention import AttentionWithEinops, AttentionWithoutEinops
from mlp import MLPWithEinops, MLPWithoutEinops
from transformer_block import TransformerBlockWithEinops, TransformerBlockWithoutEinops
from gpt import GPTWithEinops, GPTWithoutEinops
from tokenizer import (
    CharacterTokenizer,
    CharacterTokenizerWithTorch,
    BPETokenizer,
    SentencePieceTokenizer,
    TorchTokenizer,
)

device = torch.device(
    "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"
)

# Load training data
with open("training.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Initialize config
cfg = GPTConfig()

TOKENIZER_TYPE = "bpe"

if TOKENIZER_TYPE == "character":
    tokenizer = CharacterTokenizer(text)
elif TOKENIZER_TYPE == "character_torch":
    tokenizer = CharacterTokenizerWithTorch(text)
elif TOKENIZER_TYPE == "bpe":
    tokenizer = BPETokenizer(text)
elif TOKENIZER_TYPE == "sentencepiece":
    tokenizer = SentencePieceTokenizer(
        text, vocab_size=min(cfg.d_vocab, 10000))
else:
    raise ValueError(f"Unknown tokenizer type: {TOKENIZER_TYPE}")

print(f"Using {TOKENIZER_TYPE} tokenizer")
print(f"Vocabulary size: {tokenizer.vocab_size}")

sample_text = text[:100]
encoded = tokenizer.encode(sample_text)
decoded = tokenizer.decode(encoded)
print(f"\nOriginal (first 50 chars): {repr(sample_text[:50])}")
print(f"Decoded (first 50 chars):   {repr(decoded[:50])}")
print(f"Perfect match: {sample_text == decoded}")

# Update config to match tokenizer vocab size
cfg.d_vocab = tokenizer.vocab_size
print(f"Updated config d_vocab to: {cfg.d_vocab}")

# Encode entire text to tokens
data = tokenizer.encode_tensor(text)
print(f"Total tokens: {len(data)}")

# Create dataset: sequences of block_size tokens
# For each sequence, target is input shifted by 1 position
block_size = cfg.n_ctx  # Use context length from config
print(f"Block size (sequence length): {block_size}")

# Create input/target pairs


def create_dataset(data, block_size):
    """Create (input, target) pairs where target is input shifted by 1"""
    X = []
    Y = []
    for i in range(len(data) - block_size):
        X.append(data[i:i+block_size])
        Y.append(data[i+1:i+block_size+1])
    return torch.stack(X), torch.stack(Y)


X, Y = create_dataset(data, block_size)
print(f"Created {len(X)} training sequences")
print(f"Input shape: {X.shape}, Target shape: {Y.shape}")

# Split into train/val (90/10 split)
split_idx = int(0.9 * len(X))
X_train, Y_train = X[:split_idx], Y[:split_idx]
X_val, Y_val = X[split_idx:], Y[split_idx:]
print(f"Train: {len(X_train)} sequences, Val: {len(X_val)} sequences")
