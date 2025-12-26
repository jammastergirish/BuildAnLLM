# /// script
# dependencies = ["torch", "einops", "jaxtyping", "numpy", "tiktoken", "sentencepiece", "tqdm", "matplotlib"]
# ///

"""Training script for transformer models."""

import argparse
import torch
import os
from datetime import datetime
from config import ModelConfig
from pretraining.training.training_args import TransformerTrainingArgs
from pretraining.training.trainer import TransformerTrainer
from pretraining.data.dataset import TransformerDataset
from pretraining.model.model import TransformerModelWithEinops, TransformerModelWithoutEinops


def main():
    """Main function for training script."""
    parser = argparse.ArgumentParser(
        description="Train a transformer model (GPT or LLaMA)")
    parser.add_argument(
        "--architecture",
        type=str,
        default="GPT",
        choices=["GPT", "LLAMA", "OLMO"],
        help="Model architecture (GPT, LLaMA, or OLMo)"
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="small",
        choices=["small", "full"],
        help="Model size (small for faster training, full for GPT-2/LLaMA size)"
    )
    parser.add_argument(
        "--use_einops",
        action="store_true",
        help="Use einops versions of components (default: True)"
    )
    parser.add_argument(
        "--no_einops",
        dest="use_einops",
        action="store_false",
        help="Use non-einops versions of components"
    )
    parser.set_defaults(use_einops=True)
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="character",
        choices=["character", "bpe-simple", "bpe-tiktoken", "sentencepiece"],
        help="Tokenizer type (bpe-tiktoken is GPT-2 style using tiktoken)"
    )
    parser.add_argument(
        "--text_file",
        type=str,
        default="training.txt",
        help="Text file for training data"
    )

    args = parser.parse_args()

    # Device
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    # Load training data
    with open(args.text_file, "r", encoding="utf-8") as f:
        text = f.read()

    # Initialize config based on architecture and size
    if args.architecture == "LLAMA":
        if args.model_size == "small":
            cfg = ModelConfig.llama_small()
            print("Using SMALL LLaMA config (faster for Mac)")
        else:
            cfg = ModelConfig.llama_full()
            print("Using FULL LLaMA config")
    elif args.architecture == "OLMO":
        if args.model_size == "small":
            cfg = ModelConfig.olmo_small()
            print("Using SMALL OLMo config (faster for Mac)")
        else:
            cfg = ModelConfig.olmo_full()
            print("Using FULL OLMo config")
    else:  # GPT
        if args.model_size == "small":
            cfg = ModelConfig.gpt_small()
            print("Using SMALL GPT config (faster for Mac)")
        else:
            cfg = ModelConfig.gpt_full()
            print("Using FULL GPT config (GPT-2 size)")

    # Create dataset
    dataset = TransformerDataset(text, cfg, tokenizer_type=args.tokenizer_type)
    dataset.print_info()

    # Get train/val splits
    X_train, Y_train = dataset.get_train_data()
    X_val, Y_val = dataset.get_val_data()

    # Update cfg (dataset updates d_vocab internally)
    cfg = dataset.cfg

    # Initialize model
    if args.use_einops:
        model = TransformerModelWithEinops(cfg)
        model_type_str = "with_einops"
    else:
        model = TransformerModelWithoutEinops(cfg)
        model_type_str = "without_einops"

    model = model.to(device)
    print(f"\nInitialized {args.architecture} model ({model_type_str})")
    print(f"Model on device: {next(model.parameters()).device}")
    print(
        f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Training setup
    training_args = TransformerTrainingArgs()
    # Reduce eval_iters and batch_size for faster training on Mac (small models)
    if args.model_size == "small":
        training_args.eval_iters = 50  # Faster evaluation for small model
        training_args.batch_size = 16  # Smaller batch for Mac memory

    # Create timestamped checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    checkpoint_dir = os.path.join("checkpoints", timestamp)
    training_args.save_dir = checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"\nCheckpoints will be saved to: {checkpoint_dir}")

    # Create trainer
    trainer = TransformerTrainer(
        model=model,
        args=training_args,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        device=device,
        tokenizer_type=args.tokenizer_type,
    )

    # Start training
    trainer.train()

    print("\n" + "=" * 50)
    print("Training complete!")
    print("=" * 50)
    final_model_path = os.path.join(training_args.save_dir, "final_model.pt")
    print(f"Model saved to: {final_model_path}")
    print("\nTo generate text, run:")
    print(
        f"  uv run cli/infer.py --checkpoint {final_model_path} --prompt 'Your prompt here'")
    print("\nNote: Tokenizer type is automatically detected from the checkpoint.")


if __name__ == "__main__":
    main()
