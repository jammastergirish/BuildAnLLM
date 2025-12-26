"""Trainer for Supervised Fine-Tuning (SFT)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import tqdm
from finetuning.training.finetuning_args import FinetuningArgs


class SFTTrainer:
    """
    Trainer for supervised fine-tuning (SFT).

    **Key Differences from Pre-Training:**

    1. **Masked Loss**: Only computes loss on response tokens (where mask == 1),
       ignoring prompt tokens. This teaches the model to generate responses,
       not repeat prompts.

    2. **Lower Learning Rate**: Typically 10-100x lower than pre-training
       (e.g., 1e-5 vs 1e-3) to make small adjustments to pre-trained weights.

    3. **Shorter Training**: Usually 1-5 epochs vs 10+ for pre-training,
       since we're fine-tuning, not training from scratch.

    4. **Structured Data**: Trains on prompt/response pairs instead of raw text,
       teaching instruction-following behavior.

    **Training Process:**
    1. Load pre-trained model (architecture + weights)
    2. Continue training on prompt/response pairs
    3. Compute masked loss (only on response tokens)
    4. Update weights with lower learning rate
    5. Save fine-tuned checkpoints

    **Note**: This trainer reuses the same transformer architecture from pre-training.
    The model itself may use einops or not (determined by the pre-trained checkpoint).
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: FinetuningArgs,
        X_train: torch.Tensor,
        Y_train: torch.Tensor,
        masks_train: torch.Tensor,
        X_val: torch.Tensor,
        Y_val: torch.Tensor,
        masks_val: torch.Tensor,
        device: torch.device,
        eval_interval: int = 500,
        print_interval: int = 100,
        tokenizer_type: str = None,
    ):
        self.model = model
        self.args = args
        self.X_train = X_train
        self.Y_train = Y_train
        self.masks_train = masks_train
        self.X_val = X_val
        self.Y_val = Y_val
        self.masks_val = masks_val
        self.device = device
        self.eval_interval = eval_interval
        self.eval_iters = args.eval_iters
        self.print_interval = print_interval
        self.tokenizer_type = tokenizer_type
        
        # Setup optimizer (lower learning rate than pre-training)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        
        # Calculate max_iters
        self.max_iters = args.epochs * args.max_steps_per_epoch
        
        # Track running average of loss
        self.running_loss = 0.0
        self.loss_alpha = 0.99
        
        # Track losses for plotting
        self.train_losses = []
        self.val_losses = []
        self.iterations = []
        
        # Create save directory if it doesn't exist
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
    
    @torch.no_grad()
    def estimate_loss(self):
        """Estimate loss on train and val sets."""
        out = {}
        self.model.eval()
        for split_name, split_X, split_Y, split_masks in [
            ("train", self.X_train, self.Y_train, self.masks_train),
            ("val", self.X_val, self.Y_val, self.masks_val),
        ]:
            losses = torch.zeros(self.eval_iters)
            for k in tqdm(
                range(self.eval_iters),
                desc=f"Evaluating {split_name}",
                leave=False,
            ):
            # Random batch
            # idx: [batch_size] - random indices
            idx = torch.randint(0, len(split_X), (self.args.batch_size,))
            # x_batch: [batch_size, seq_len] - input sequences
            # y_batch: [batch_size, seq_len] - target sequences
            # masks_batch: [batch_size, seq_len] - loss masks (1 for response, 0 for prompt)
            x_batch = split_X[idx].to(self.device)
            y_batch = split_Y[idx].to(self.device)
            masks_batch = split_masks[idx].to(self.device)

            # Forward pass
            # logits: [batch_size, seq_len, vocab_size] - model predictions
            logits = self.model(x_batch)

            # Compute loss only on response tokens (where mask == 1)
            # This is the key difference from pre-training: we ignore loss on prompt tokens
            # Reshape for cross_entropy
            # logits_flat: [batch_size * seq_len, vocab_size]
            # targets_flat: [batch_size * seq_len]
            # masks_flat: [batch_size * seq_len] - 1 for response, 0 for prompt
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = y_batch.view(-1)
            masks_flat = masks_batch.view(-1)

            # Cross-entropy loss (per token, before masking)
            # loss_unmasked: [batch_size * seq_len] - loss for each position
            loss_unmasked = F.cross_entropy(
                logits_flat, targets_flat, reduction='none'
            )
            # Apply mask: only compute loss on response tokens
            # loss_masked: scalar - average loss over response tokens only
            loss_masked = (loss_unmasked * masks_flat).sum() / masks_flat.sum().clamp(min=1)
            losses[k] = loss_masked.item()
            out[split_name] = losses.mean().item()
        self.model.train()
        return out
    
    def train(self):
        """Main training loop."""
        print("\nStarting fine-tuning...")
        print(f"Training for {self.args.epochs} epochs, {self.max_iters} total iterations")
        print(f"Batch size: {self.args.batch_size}, Learning rate: {self.args.lr}")
        print(f"Weight decay: {self.args.weight_decay}")
        print(f"Evaluating every {self.eval_interval} iterations\n")
        
        pbar = tqdm(range(self.max_iters), desc="Fine-tuning")
        for iter_num in pbar:
            # Get random batch
            # idx: [batch_size] - random indices into training set
            idx = torch.randint(0, len(self.X_train), (self.args.batch_size,))
            # x_batch: [batch_size, seq_len] - input token sequences
            # y_batch: [batch_size, seq_len] - target token sequences (shifted by 1)
            # masks_batch: [batch_size, seq_len] - loss masks (1 for response, 0 for prompt)
            x_batch = self.X_train[idx].to(self.device)
            y_batch = self.Y_train[idx].to(self.device)
            masks_batch = self.masks_train[idx].to(self.device)

            # Forward pass
            # logits: [batch_size, seq_len, vocab_size] - model predictions
            logits = self.model(x_batch)

            # Compute masked loss (only on response tokens)
            # Key difference from pre-training: we ignore loss on prompt tokens
            # This teaches the model to generate responses, not repeat prompts
            # Reshape for cross_entropy
            # logits_flat: [batch_size * seq_len, vocab_size]
            # targets_flat: [batch_size * seq_len]
            # masks_flat: [batch_size * seq_len] - 1 for response, 0 for prompt
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = y_batch.view(-1)
            masks_flat = masks_batch.view(-1)

            # Cross-entropy loss per token (before masking)
            # loss_unmasked: [batch_size * seq_len] - loss for each position
            loss_unmasked = F.cross_entropy(
                logits_flat, targets_flat, reduction='none'
            )
            # Apply mask: multiply by mask (0 for prompt, 1 for response), then average
            # loss: scalar - average loss over response tokens only
            loss = (loss_unmasked * masks_flat).sum() / masks_flat.sum().clamp(min=1)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update running loss
            self.running_loss = (
                self.loss_alpha * self.running_loss
                + (1 - self.loss_alpha) * loss.item()
            )
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{self.running_loss:.4f}",
            })
            
            # Print detailed loss periodically
            if iter_num % self.print_interval == 0 and iter_num > 0:
                print(
                    f"\n[Iter {iter_num}] Current loss: {loss.item():.4f}, "
                    f"Running avg: {self.running_loss:.4f}"
                )
            
            # Evaluate at intervals
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
                self.iterations.append(iter_num)
                self.train_losses.append(losses['train'])
                self.val_losses.append(losses['val'])
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{self.running_loss:.4f}",
                    "val_loss": f"{losses['val']:.4f}",
                })
            
            # Save checkpoint
            if (
                hasattr(self.args, "save_interval")
                and iter_num % self.args.save_interval == 0
                and iter_num > 0
            ):
                self.save_checkpoint(iter_num)
        
        pbar.close()
        print("\nFine-tuning complete!")
        print(f"Final running average loss: {self.running_loss:.4f}")
        
        # Save final checkpoint
        if self.args.save_dir:
            self.save_checkpoint(self.max_iters, is_final=True)
    
    def save_checkpoint(self, iter_num: int, is_final: bool = False):
        """Save model checkpoint."""
        if not self.args.save_dir:
            return
        
        if is_final:
            filepath = os.path.join(self.args.save_dir, "final_model.pt")
        else:
            filepath = os.path.join(
                self.args.save_dir, f"checkpoint_{iter_num}.pt"
            )
        
        # Save model config
        cfg = self.model.cfg if hasattr(self.model, "cfg") else None
        cfg_dict = cfg.to_dict() if cfg is not None else None
        
        # Determine model type
        model_type = "with_einops" if "WithEinops" in self.model.__class__.__name__ else "without_einops"
        
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "iter_num": iter_num,
                "running_loss": self.running_loss,
                "args": self.args,
                "cfg": cfg_dict,
                "model_type": model_type,
                "tokenizer_type": self.tokenizer_type,
                "is_finetuned": True,  # Mark as fine-tuned
            },
            filepath,
        )
        if not is_final:
            print(f"Checkpoint saved: {filepath}")

