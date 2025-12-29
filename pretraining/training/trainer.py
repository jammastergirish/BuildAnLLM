import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from pretraining.training.training_args import TransformerTrainingArgs
from pretraining.utils import extract_model_output_and_aux_loss, add_aux_loss_to_main_loss


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
        tokenizer_type: str = None,
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
        self.tokenizer_type = tokenizer_type

        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # Calculate max_iters
        self.max_iters = args.epochs * args.max_steps_per_epoch

        # Track running average of loss
        self.running_loss = 0.0
        self.loss_alpha = 0.99  # Exponential moving average

        # Track losses for plotting
        self.train_losses = []
        self.val_losses = []
        self.iterations = []

        # Create save directory if it doesn't exist
        if hasattr(args, "save_dir"):
            os.makedirs(args.save_dir, exist_ok=True)

    def _evaluate_batch(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor
    ) -> float:
        """Compute loss for a single batch.

        Args:
            x_batch: Input tokens [batch_size, seq_len]
            y_batch: Target tokens [batch_size, seq_len]

        Returns:
            Loss value (scalar float)
        """
        model_output = self.model(x_batch)
        logits, aux_loss = extract_model_output_and_aux_loss(model_output)

        # Reshape for cross_entropy: (batch*seq, vocab) and (batch*seq,)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y_batch.view(-1)
        )
        loss = add_aux_loss_to_main_loss(loss, aux_loss, self.model)
        return loss.item()

    def _evaluate_split(
        self,
        split_name: str,
        split_X: torch.Tensor,
        split_Y: torch.Tensor
    ) -> float:
        """Evaluate one split (train or val).

        Computes average loss over multiple random batches from the split.

        Args:
            split_name: Name of split ("train" or "val")
            split_X: Input tokens for split
            split_Y: Target tokens for split

        Returns:
            Average loss over evaluation iterations
        """
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

            losses[k] = self._evaluate_batch(x_batch, y_batch)

        return losses.mean().item()

    @torch.no_grad()
    def estimate_loss(self):
        """Estimate loss on train and val sets."""
        out = {}
        self.model.eval()

        # Evaluate train split
        out["train"] = self._evaluate_split(
            "train", self.X_train, self.Y_train)

        # Evaluate val split
        out["val"] = self._evaluate_split("val", self.X_val, self.Y_val)

        self.model.train()
        return out

    def _training_step(
        self,
        x_batch: torch.Tensor,
        y_batch: torch.Tensor
    ) -> torch.Tensor:
        """Perform one training step.

        Handles forward pass, loss computation (including aux_loss from MoE),
        backward pass, and optimizer step.

        Args:
            x_batch: Input tokens [batch_size, seq_len]
            y_batch: Target tokens [batch_size, seq_len]

        Returns:
            Loss tensor (scalar)
        """
        # Forward pass
        # logits: [batch_size, seq_len, vocab_size]
        # May return (logits, aux_loss) if MoE is enabled
        model_output = self.model(x_batch)
        logits, aux_loss = extract_model_output_and_aux_loss(model_output)

        # Reshape for cross_entropy
        # logits.view(-1, logits.size(-1)): [batch_size * seq_len, vocab_size]
        # y_batch.view(-1): [batch_size * seq_len]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y_batch.view(-1)
        )
        loss = add_aux_loss_to_main_loss(loss, aux_loss, self.model)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

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
            # idx: [batch_size]
            idx = torch.randint(0, len(self.X_train), (self.args.batch_size,))
            # x_batch: [batch_size, seq_len]
            # y_batch: [batch_size, seq_len]
            x_batch = self.X_train[idx].to(self.device)
            y_batch = self.Y_train[idx].to(self.device)

            # Perform training step
            loss = self._training_step(x_batch, y_batch)

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
                # Track losses for plotting
                self.iterations.append(iter_num)
                self.train_losses.append(losses['train'])
                self.val_losses.append(losses['val'])
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
            # Save loss graph
            self.save_loss_graph()

    def train_single_step(self):
        """Perform a single training step (one batch) and return metrics.
        
        Designed for interactive 'Stepping' in UI.
        
        Returns:
            dict with metrics: 'loss', 'grad_norm'
        """
        # Get random batch
        idx = torch.randint(0, len(self.X_train), (self.args.batch_size,))
        x_batch = self.X_train[idx].to(self.device)
        y_batch = self.Y_train[idx].to(self.device)

        # Forward pass
        model_output = self.model(x_batch)
        logits, aux_loss = extract_model_output_and_aux_loss(model_output)

        # Compute loss
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), y_batch.view(-1)
        )
        loss = add_aux_loss_to_main_loss(loss, aux_loss, self.model)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Calculate gradient norm (educational metric)
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.optimizer.step()
        
        # Update running loss
        self.running_loss = (
            self.loss_alpha * self.running_loss
            + (1 - self.loss_alpha) * loss.item()
        )
        
        return {
            "loss": loss.item(),
            "running_loss": self.running_loss,
            "grad_norm": total_norm,
            "aux_loss": aux_loss.item() if aux_loss is not None else 0.0,
            "inputs": x_batch.detach().cpu(), 
            "targets": y_batch.detach().cpu()
        }

    def save_checkpoint(self, iter_num: int, is_final: bool = False):
        """Save model checkpoint"""
        if not hasattr(self.args, "save_dir"):
            return

        if is_final:
            filepath = os.path.join(self.args.save_dir, "final_model.pt")
        else:
            filepath = os.path.join(
                self.args.save_dir, f"checkpoint_{iter_num}.pt")

        # Save model config from model as dict for proper serialization
        cfg = self.model.cfg if hasattr(self.model, "cfg") else None
        cfg_dict = cfg.to_dict() if cfg is not None else None

        # Determine model type from use_einops attribute (if available)
        # Otherwise check class name for backward compatibility
        if hasattr(self.model, "use_einops"):
            model_type = "with_einops" if self.model.use_einops else "without_einops"
        else:
            model_type = "with_einops" if "WithEinops" in self.model.__class__.__name__ else "without_einops"

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iter_num": iter_num,
                "running_loss": self.running_loss,
                "args": self.args,
                "cfg": cfg_dict,  # Save as dict for proper enum serialization
                "model_type": model_type,
                "tokenizer_type": self.tokenizer_type,
            },
            filepath,
        )
        if not is_final:
            print(f"Checkpoint saved: {filepath}")

    def save_loss_graph(self):
        """Save loss graph to checkpoint directory"""
        if not hasattr(self.args, "save_dir") or not self.iterations:
            return

        plt.figure(figsize=(10, 6))
        plt.plot(self.iterations, self.train_losses,
                 label="Train Loss", marker="o", markersize=3)
        plt.plot(self.iterations, self.val_losses,
                 label="Val Loss", marker="s", markersize=3)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        graph_path = os.path.join(self.args.save_dir, "loss_graph.png")
        plt.savefig(graph_path, dpi=150)
        plt.close()
        print(f"Loss graph saved to: {graph_path}")

    @staticmethod
    def load_checkpoint(
        filepath: str, model: nn.Module, optimizer: torch.optim.Optimizer = None
    ):
        """Load model checkpoint"""
        # Allowlist TransformerTrainingArgs for safe loading (PyTorch 2.6+)
        torch.serialization.add_safe_globals([TransformerTrainingArgs])
        checkpoint = torch.load(
            filepath, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Checkpoint loaded from {filepath}")
        return checkpoint
