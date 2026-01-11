"""Tokenizer helpers shared across endpoints."""

from __future__ import annotations

from pathlib import Path

from pretraining.tokenization.tokenizer import (
    BPETokenizer,
    CharacterTokenizer,
    SentencePieceTokenizer,
    SimpleBPETokenizer,
)


class TokenizerLoadError(RuntimeError):
    pass


def load_tokenizer_for_checkpoint(
    checkpoint_path: Path,
    tokenizer_type: str,
    vocab_size: int,
    training_text_path: Path = Path("input_data/pretraining/orwell.txt"),
):
    if tokenizer_type == "character":
        text = _read_training_text(training_text_path)
        return CharacterTokenizer(text)
    if tokenizer_type == "bpe-simple":
        model_path = checkpoint_path.parent / "tokenizer.model"
        if model_path.exists():
            return SimpleBPETokenizer(model_path=str(model_path))
        text = _read_training_text(training_text_path)
        return SimpleBPETokenizer(text, vocab_size=min(vocab_size or 1000, 5000))
    if tokenizer_type in {"bpe-tiktoken", "bpe"}:
        return BPETokenizer()
    if tokenizer_type == "sentencepiece":
        model_path = checkpoint_path.parent / "tokenizer.model"
        if model_path.exists():
            return SentencePieceTokenizer(model_path=str(model_path))
        text = _read_training_text(training_text_path)
        return SentencePieceTokenizer(text, vocab_size=min(vocab_size or 10000, 10000))
    raise TokenizerLoadError(f"Unknown tokenizer type: {tokenizer_type}")


def _read_training_text(path: Path) -> str:
    if not path.exists():
        raise TokenizerLoadError(
            f"Training text file not found at {path}. Provide a file or tokenizer model."
        )
    return path.read_text(encoding="utf-8")
