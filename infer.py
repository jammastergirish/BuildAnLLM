# /// script
# dependencies = ["torch", "einops", "jaxtyping", "numpy", "tiktoken", "sentencepiece", "tqdm"]
# ///

"""Inference script for generating text from trained GPT models."""

import argparse
from tokenizer import CharacterTokenizer, BPETokenizer
from sampler import TransformerSampler
from model_loader import load_model_from_checkpoint
from utils import get_device


def main():
    """Main function for inference script."""
    parser = argparse.ArgumentParser(
        description="Generate text from trained GPT model")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/final_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="First Citizen:",
        help="Starting prompt for text generation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (higher = more random)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Top-k sampling (None to disable)"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling (None to disable)"
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default=None,
        choices=["character", "bpe"],
        help="Tokenizer type (auto-detected from checkpoint if not provided)"
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default="training.txt",
        help="Text file for tokenizer initialization (needed for character tokenizer)"
    )

    args = parser.parse_args()

    # Device
    device = get_device()
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint from {args.checkpoint}...")
    model, _, checkpoint = load_model_from_checkpoint(args.checkpoint, device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    model_type = checkpoint.get("model_type", "with_einops")
    print(f"Model loaded: {model_type}, {param_count:.2f}M parameters")

    # Get tokenizer type from checkpoint or args
    tokenizer_type = checkpoint.get("tokenizer_type")
    if tokenizer_type is None:
        # Fallback to command-line argument
        if args.tokenizer_type is None:
            raise ValueError(
                "Tokenizer type not found in checkpoint and not provided. "
                "Please specify --tokenizer_type (character or bpe)"
            )
        tokenizer_type = args.tokenizer_type
        print(
            f"Warning: Tokenizer type not in checkpoint, using provided: {tokenizer_type}")
    else:
        # Check if user provided a different tokenizer type
        if args.tokenizer_type is not None and args.tokenizer_type != tokenizer_type:
            print(
                f"Warning: Checkpoint uses tokenizer type '{tokenizer_type}', "
                f"but you provided '{args.tokenizer_type}'. "
                f"Using '{tokenizer_type}' from checkpoint."
            )

    # Create tokenizer (must match training tokenizer)
    if tokenizer_type == "character":
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
        tokenizer = CharacterTokenizer(text)
    elif tokenizer_type == "bpe":
        tokenizer = BPETokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")

    print(
        f"Using {tokenizer_type} tokenizer (vocab size: {tokenizer.vocab_size})")

    # Create sampler
    sampler = TransformerSampler(
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    # Generate text
    print("\n" + "=" * 50)
    print("Generating text...")
    print("=" * 50)
    print(f"Prompt: {args.prompt}")
    print(
        f"Temperature: {args.temperature}, Top-k: {args.top_k}, Top-p: {args.top_p}")
    print("-" * 50)

    generated = sampler.sample(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )

    print(f"\nGenerated text:\n{generated}")
    print("=" * 50)


if __name__ == "__main__":
    main()
