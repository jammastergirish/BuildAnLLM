"""Train or fine-tune a small transformer language model with PyTorch.

Examples:
    python examples/simple_transformer.py train --data input_data/pretraining/wilde.txt
    python examples/simple_transformer.py finetune \
        --data my_domain_text.txt \
        --checkpoint checkpoints/simple_transformer.pt \
        --output checkpoints/fine_tuned.pt
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB_SIZE = 256  # One token for each possible UTF-8 byte.


@dataclass
class ModelConfig:
    context_length: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1


class TransformerLanguageModel(nn.Module):
    """A small decoder-only transformer for next-token prediction."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB_SIZE, config.d_model)
        self.position_embedding = nn.Embedding(
            config.context_length, config.d_model
        )

        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=4 * config.d_model,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, config.n_layers)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output = nn.Linear(config.d_model, VOCAB_SIZE, bias=False)

        # Input and output tokens share one weight matrix, as in many LLMs.
        self.output.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.size(1)
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds the model context "
                f"length {self.config.context_length}."
            )

        positions = torch.arange(sequence_length, device=token_ids.device)
        hidden = self.token_embedding(token_ids)
        hidden = hidden + self.position_embedding(positions)

        # True entries are hidden from attention, preventing access to the future.
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=token_ids.device,
            ),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask)
        return self.output(self.final_norm(hidden))


def encode(text: str) -> torch.Tensor:
    return torch.tensor(list(text.encode("utf-8")), dtype=torch.long)


def decode(token_ids: torch.Tensor) -> str:
    return bytes(token_ids.tolist()).decode("utf-8", errors="replace")


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def random_batch(
    tokens: torch.Tensor,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) <= context_length:
        raise ValueError(
            f"Training text needs more than {context_length} bytes; got {len(tokens)}."
        )

    starts = torch.randint(
        len(tokens) - context_length,
        size=(batch_size,),
    )
    inputs = torch.stack([tokens[i : i + context_length] for i in starts])
    targets = torch.stack([tokens[i + 1 : i + context_length + 1] for i in starts])
    return inputs.to(device), targets.to(device)


def train(
    model: TransformerLanguageModel,
    tokens: torch.Tensor,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device,
    log_every: int,
) -> None:
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for step in range(1, steps + 1):
        inputs, targets = random_batch(
            tokens, batch_size, model.config.context_length, device
        )
        logits = model(inputs)
        loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == steps:
            print(f"step {step:>5}/{steps} | loss {loss.item():.4f}")


@torch.no_grad()
def generate(
    model: TransformerLanguageModel,
    prompt: str,
    new_tokens: int,
    temperature: float,
    device: torch.device,
) -> str:
    model.eval()
    token_ids = encode(prompt).unsqueeze(0).to(device)

    for _ in range(new_tokens):
        context = token_ids[:, -model.config.context_length :]
        next_token_logits = model(context)[:, -1]

        if temperature == 0:
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
        else:
            probabilities = F.softmax(next_token_logits / temperature, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)

        token_ids = torch.cat((token_ids, next_token), dim=1)

    return decode(token_ids[0].cpu())


def save_checkpoint(
    path: Path,
    model: TransformerLanguageModel,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.config),
            "model_state": model.state_dict(),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> TransformerLanguageModel:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = TransformerLanguageModel(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a new model")
    train_parser.add_argument("--data", type=Path, required=True)
    train_parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/simple_transformer.pt")
    )
    train_parser.add_argument("--context-length", type=int, default=64)
    train_parser.add_argument("--d-model", type=int, default=128)
    train_parser.add_argument("--n-heads", type=int, default=4)
    train_parser.add_argument("--n-layers", type=int, default=2)

    finetune_parser = subparsers.add_parser(
        "finetune", help="Continue training an existing model"
    )
    finetune_parser.add_argument("--data", type=Path, required=True)
    finetune_parser.add_argument("--checkpoint", type=Path, required=True)
    finetune_parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/fine_tuned.pt")
    )

    for subparser in (train_parser, finetune_parser):
        subparser.add_argument("--steps", type=int, default=500)
        subparser.add_argument("--batch-size", type=int, default=32)
        subparser.add_argument("--learning-rate", type=float, default=None)
        subparser.add_argument("--log-every", type=int, default=50)
        subparser.add_argument("--prompt", default="The ")
        subparser.add_argument("--new-tokens", type=int, default=100)
        subparser.add_argument("--temperature", type=float, default=0.8)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative.")

    torch.manual_seed(42)
    device = choose_device()
    text = args.data.read_text(encoding="utf-8")
    tokens = encode(text)

    if args.command == "train":
        config = ModelConfig(
            context_length=args.context_length,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
        )
        model = TransformerLanguageModel(config).to(device)
        learning_rate = args.learning_rate or 3e-4
    else:
        model = load_checkpoint(args.checkpoint, device)
        learning_rate = args.learning_rate or 3e-5

    print(f"device: {device}")
    train(
        model,
        tokens,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=learning_rate,
        device=device,
        log_every=args.log_every,
    )
    save_checkpoint(args.output, model)
    print(f"saved: {args.output}")
    print(generate(model, args.prompt, args.new_tokens, args.temperature, device))


if __name__ == "__main__":
    main()
