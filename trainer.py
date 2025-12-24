import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from training_args import TransformerTrainingArgs


class TransformerTrainer:
    def __init__(
        self,
        model: nn.Module,
        args: TransformerTrainingArgs,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        device: torch.device,
        eval_interval: int = 500,
        print_interval: int = 100,
    ):
        self.model = model
        self.args = args
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.device = device
        self.eval_interval = eval_interval
        self.eval_iters = getattr(args, "eval_iters", 200)
        self.print_interval = print_interval

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # Calculate max_iters
        self.max_iters = args.epochs * args.max_steps_per_epoch

        # Track running average of loss
        self.running_loss = 0.0
        self.loss_alpha = 0.99  # Exponential moving average

        # Create save directory if it doesn't exist
        if hasattr(args, "save_dir"):
            os.makedirs(args.save_dir, exist_ok=True)

    @torch.no_grad()
    def estimate_loss(self):
        """Estimate loss on train and val sets"""
        out = {}
        self.model.eval()
        for split_name, split_X, split_Y in [
            ("train", self.X_train, self.Y_train),
            ("val", self.X_val, self.Y_val),
        ]:
            losses = torch.zeros(self.eval_iters)
            for k in tqdm(
                range(self.eval_iters),
                desc=f"Evaluating {split_name}",
                leave=False,
            ):
                # Random batch
                idx = torch.randint(0, len(split_X), (self.args.batch_size,))
                x_batch = split_X[idx].to(self.device)
                y_batch = split_Y[idx].to(self.device)

                logits = self.model(x_batch)
                # Reshape for cross_entropy: (batch*seq, vocab) and (batch*seq,)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y_batch.view(-1)
                )
                losses[k] = loss.item()
            out[split_name] = losses.mean().item()
        self.model.train()
        return out

    def train(self):
        """Main training loop"""
        print("\nStarting training...")
        print(
            f"Training for {self.args.epochs} epochs, {self.max_iters} total iterations"
        )
        print(
            f"Batch size: {self.args.batch_size}, Learning rate: {self.args.lr}")
        print(f"Weight decay: {self.args.weight_decay}")
        print(f"Evaluating every {self.eval_interval} iterations\n")

        pbar = tqdm(range(self.max_iters), desc="Training")
        for iter_num in pbar:
            # Get random batch
            idx = torch.randint(0, len(self.X_train), (self.args.batch_size,))
            x_batch = self.X_train[idx].to(self.device)
            y_batch = self.Y_train[idx].to(self.device)

            # Forward pass
            logits = self.model(x_batch)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y_batch.view(-1)
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update running loss
            self.running_loss = (
                self.loss_alpha * self.running_loss
                + (1 - self.loss_alpha) * loss.item()
            )

            # Update progress bar with current loss
            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{self.running_loss:.4f}",
                }
            )

            # Print detailed loss periodically
            if iter_num % self.print_interval == 0 and iter_num > 0:
                print(
                    f"\n[Iter {iter_num}] Current loss: {loss.item():.4f}, "
                    f"Running avg: {self.running_loss:.4f}"
                )

            # Evaluate at intervals and final iteration
            should_eval = (
                (iter_num > 0 and iter_num % self.eval_interval == 0)
                or iter_num == self.max_iters - 1
            )
            if should_eval:
                losses = self.estimate_loss()
                print(
                    f"\n[Iter {iter_num}] Train loss: {losses['train']:.4f}, "
                    f"Val loss: {losses['val']:.4f}"
                )
                # Update progress bar with eval metrics
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "avg_loss": f"{self.running_loss:.4f}",
                        "val_loss": f"{losses['val']:.4f}",
                    }
                )

            # Save checkpoint
            if (
                hasattr(self.args, "save_interval")
                and iter_num % self.args.save_interval == 0
                and iter_num > 0
            ):
                self.save_checkpoint(iter_num)

        pbar.close()
        print("\nTraining complete!")
        print(f"Final running average loss: {self.running_loss:.4f}")

        # Save final checkpoint
        if hasattr(self.args, "save_dir"):
            self.save_checkpoint(self.max_iters, is_final=True)

    def save_checkpoint(self, iter_num: int, is_final: bool = False):
        """Save model checkpoint"""
        if not hasattr(self.args, "save_dir"):
            return

        if is_final:
            filepath = os.path.join(self.args.save_dir, "final_model.pt")
        else:
            filepath = os.path.join(
                self.args.save_dir, f"checkpoint_{iter_num}.pt")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iter_num": iter_num,
                "running_loss": self.running_loss,
                "args": self.args,
            },
            filepath,
        )
        if not is_final:
            print(f"Checkpoint saved: {filepath}")

    @staticmethod
    def load_checkpoint(
        filepath: str, model: nn.Module, optimizer: torch.optim.Optimizer = None
    ):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint
