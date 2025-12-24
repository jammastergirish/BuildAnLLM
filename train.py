# /// script
# dependencies = ["torch", "einops", "jaxtyping", "numpy", "tiktoken", "sentencepiece", "tqdm"]
# ///

import torch
from config import GPTConfig
from training_args import TransformerTrainingArgs
from trainer import TransformerTrainer
from dataset import TransformerDataset
from gpt import GPTWithEinops, GPTWithoutEinops
from sampler import TransformerSampler

device = torch.device(
    "mps" if torch.backends.mps.is_available(
    ) else "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"Using device: {device}")

# Load training data
with open("training.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Initialize config
# Use GPTConfig.small() for faster training on Mac, GPTConfig() for full model
USE_SMALL_MODEL = True  # Set to False for full GPT-2 size model

if USE_SMALL_MODEL:
    cfg = GPTConfig.small()
    print("Using SMALL model config (faster for Mac)")
else:
    cfg = GPTConfig()
    print("Using FULL model config (GPT-2 size)")

# Create dataset
TOKENIZER_TYPE = "bpe"
dataset = TransformerDataset(text, cfg, tokenizer_type=TOKENIZER_TYPE)
dataset.print_info()

# Get train/val splits
X_train, Y_train = dataset.get_train_data()
X_val, Y_val = dataset.get_val_data()

# Update cfg (dataset updates d_vocab internally)
cfg = dataset.cfg

# Initialize model
MODEL_TYPE = "with_einops"  # Options: "with_einops", "without_einops"

if MODEL_TYPE == "with_einops":
    model = GPTWithEinops(cfg)
else:
    model = GPTWithoutEinops(cfg)

model = model.to(device)
print(f"\nInitialized {MODEL_TYPE} model")
print(f"Model on device: {next(model.parameters()).device}")
print(
    f"Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# Training setup
args = TransformerTrainingArgs()
# Reduce eval_iters and batch_size for faster training on Mac
if USE_SMALL_MODEL:
    args.eval_iters = 50  # Faster evaluation for small model
    args.batch_size = 16  # Smaller batch for Mac memory

# Create trainer
trainer = TransformerTrainer(
    model=model,
    args=args,
    X_train=X_train,
    Y_train=Y_train,
    X_val=X_val,
    Y_val=Y_val,
    device=device,
)

# Start training
trainer.train()

# After training, create sampler for text generation
print("\n" + "=" * 50)
print("Training complete! Creating sampler for text generation...")
sampler = TransformerSampler(
    model=model, tokenizer=dataset.tokenizer, device=device)

# Sample with different strategies
print("\n" + "=" * 50)
print("Generating text samples with different strategies...")
print("=" * 50)

prompt = "First Citizen:"

# 1. Low temperature (deterministic, focused)
print("\n1. Low temperature (0.5, focused):")
generated = sampler.sample(
    prompt, max_new_tokens=200, temperature=0.5, top_k=None, top_p=None
)
print(f"Prompt: {prompt}")
print(f"Generated:\n{generated}\n")

# 2. Medium temperature
print("2. Medium temperature (1.0, balanced):")
generated = sampler.sample(
    prompt, max_new_tokens=200, temperature=1.0, top_k=None, top_p=None
)
print(f"Prompt: {prompt}")
print(f"Generated:\n{generated}\n")

# 3. High temperature (more creative)
print("3. High temperature (1.5, creative):")
generated = sampler.sample(
    prompt, max_new_tokens=200, temperature=1.5, top_k=None, top_p=None
)
print(f"Prompt: {prompt}")
print(f"Generated:\n{generated}\n")

# 4. Top-k sampling
print("4. Top-k sampling (k=40, temperature=0.8):")
generated = sampler.sample(
    prompt, max_new_tokens=200, temperature=0.8, top_k=40, top_p=None
)
print(f"Prompt: {prompt}")
print(f"Generated:\n{generated}\n")

# 5. Top-p (nucleus) sampling
print("5. Top-p (nucleus) sampling (p=0.9, temperature=0.8):")
generated = sampler.sample(
    prompt, max_new_tokens=200, temperature=0.8, top_k=None, top_p=0.9
)
print(f"Prompt: {prompt}")
print(f"Generated:\n{generated}\n")

# 6. Combined top-k and top-p
print("6. Combined top-k and top-p (k=40, p=0.9, temperature=0.8):")
generated = sampler.sample(
    prompt, max_new_tokens=200, temperature=0.8, top_k=40, top_p=0.9
)
print(f"Prompt: {prompt}")
print(f"Generated:\n{generated}\n")

print("=" * 50)
print("Sampling complete!")
