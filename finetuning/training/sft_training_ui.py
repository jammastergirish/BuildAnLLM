"""Training UI components for SFT fine-tuning."""

import streamlit as st
import torch
import threading
from typing import Dict, Any
from collections import deque
from tqdm import tqdm

from finetuning.training.sft_trainer import SFTTrainer


def train_sft_model_thread(
    trainer: SFTTrainer,
    shared_loss_data: Dict[str, list],
    shared_logs: deque,
    training_active_flag: list,
    lock: threading.Lock,
    progress_data: Dict[str, Any]
) -> None:
    """Training thread for SFT that updates shared data structures (thread-safe)."""
    try:
        max_iters = trainer.max_iters
        eval_interval = trainer.eval_interval
        print_interval = getattr(trainer, 'print_interval', 100)

        _log_training_start(trainer, shared_logs, lock, eval_interval)

        pbar = tqdm(range(max_iters), desc="Fine-tuning")
        first_loss_set = False

        for iter_num in pbar:
            if not _check_training_active(training_active_flag, lock, shared_logs):
                break

            # Training step (with masks)
            loss, running_loss = _sft_training_step(
                trainer, iter_num, first_loss_set)
            if not first_loss_set:
                trainer.running_loss = loss.item()
                first_loss_set = True
            else:
                trainer.running_loss = running_loss

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{trainer.running_loss:.4f}",
            })

            _update_progress(progress_data, iter_num, loss.item(),
                             trainer.running_loss, max_iters, lock)

            if iter_num % print_interval == 0 and iter_num > 0:
                _log_iteration(iter_num, loss.item(), trainer.running_loss,
                               shared_logs, lock)

            if (iter_num > 0 and iter_num % eval_interval == 0) or iter_num == max_iters - 1:
                _evaluate_and_log(trainer, iter_num, shared_loss_data,
                                  shared_logs, progress_data, lock, pbar)

            if (hasattr(trainer.args, "save_interval") and
                    iter_num % trainer.args.save_interval == 0 and iter_num > 0):
                trainer.save_checkpoint(iter_num)
                _log_checkpoint_saved(iter_num, shared_logs, lock)

        pbar.close()
        _finalize_training(trainer, max_iters, training_active_flag,
                           shared_logs, progress_data, lock)

    except Exception as e:
        import traceback
        with lock:
            shared_logs.append(f"Error during fine-tuning: {str(e)}")
            shared_logs.append(traceback.format_exc())
            training_active_flag[0] = False


def _log_training_start(trainer, shared_logs, lock, eval_interval):
    """Log training start information."""
    print("\nStarting fine-tuning...")
    print(
        f"Training for {trainer.args.epochs} epochs, {trainer.max_iters} total iterations")
    print(
        f"Batch size: {trainer.args.batch_size}, Learning rate: {trainer.args.lr}")
    print(f"Weight decay: {trainer.args.weight_decay}")
    print(f"Evaluating every {eval_interval} iterations\n")

    with lock:
        shared_logs.extend([
            "Starting fine-tuning...",
            f"Training for {trainer.args.epochs} epochs, {trainer.max_iters} total iterations",
            f"Batch size: {trainer.args.batch_size}, Learning rate: {trainer.args.lr}",
            f"Weight decay: {trainer.args.weight_decay}",
            f"Evaluating every {eval_interval} iterations\n"
        ])


def _check_training_active(flag, lock, logs):
    """Check if training should continue."""
    with lock:
        if not flag[0]:
            logs.append("Fine-tuning stopped by user.")
            return False
    return True


def _sft_training_step(trainer, iter_num, first_loss_set):
    """Perform one SFT training step with masks."""
    idx = torch.randint(0, len(trainer.X_train), (trainer.args.batch_size,))
    x_batch = trainer.X_train[idx].to(trainer.device)
    y_batch = trainer.Y_train[idx].to(trainer.device)
    masks_batch = trainer.masks_train[idx].to(trainer.device)

    logits = trainer.model(x_batch)
    
    # Compute masked loss (only on response tokens)
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = y_batch.view(-1)
    masks_flat = masks_batch.view(-1)
    
    loss_unmasked = torch.nn.functional.cross_entropy(
        logits_flat, targets_flat, reduction='none'
    )
    loss = (loss_unmasked * masks_flat).sum() / masks_flat.sum().clamp(min=1)

    trainer.optimizer.zero_grad()
    loss.backward()
    trainer.optimizer.step()

    if not first_loss_set:
        running_loss = loss.item()
    else:
        running_loss = (trainer.loss_alpha * trainer.running_loss +
                        (1 - trainer.loss_alpha) * loss.item())

    return loss, running_loss


def _update_progress(progress_data, iter_num, loss, running_loss, max_iters, lock):
    """Update progress data."""
    should_update = (iter_num % 10 == 0 or iter_num ==
                     max_iters - 1 or iter_num == 0)
    if should_update:
        with lock:
            progress_data["iter"] = iter_num
            progress_data["loss"] = loss
            progress_data["running_loss"] = running_loss
            progress_data["progress"] = min((iter_num + 1) / max_iters, 1.0)
            if "all_losses" in progress_data:
                progress_data["all_losses"]["iterations"].append(iter_num)
                progress_data["all_losses"]["current_losses"].append(loss)
                progress_data["all_losses"]["running_losses"].append(
                    running_loss)


def _log_iteration(iter_num, loss, running_loss, shared_logs, lock):
    """Log iteration details."""
    msg = f"\n[Iter {iter_num}] Current loss: {loss:.4f}, Running avg: {running_loss:.4f}"
    print(msg)
    with lock:
        shared_logs.append(msg)


def _evaluate_and_log(trainer, iter_num, shared_loss_data, shared_logs,
                      progress_data, lock, pbar):
    """Evaluate and log results."""
    losses = trainer.estimate_loss()
    print(f"\n[Iter {iter_num}] Train loss: {losses['train']:.4f}, "
          f"Val loss: {losses['val']:.4f}")
    pbar.set_postfix({
        "loss": f"{losses['train']:.4f}",
        "avg_loss": f"{trainer.running_loss:.4f}",
        "val_loss": f"{losses['val']:.4f}",
    })
    with lock:
        shared_loss_data["iterations"].append(iter_num)
        shared_loss_data["train_losses"].append(losses['train'])
        shared_loss_data["val_losses"].append(losses['val'])
        shared_logs.append(
            f"[Iter {iter_num}] Train loss: {losses['train']:.4f}, "
            f"Val loss: {losses['val']:.4f}"
        )
        progress_data["val_loss"] = losses['val']


def _log_checkpoint_saved(iter_num, shared_logs, lock):
    """Log checkpoint save."""
    msg = f"Checkpoint saved at iteration {iter_num}"
    print(msg)
    with lock:
        shared_logs.append(msg)


def _finalize_training(trainer, max_iters, training_active_flag,
                       shared_logs, progress_data, lock):
    """Finalize training and save results."""
    print("\nFine-tuning complete!")
    print(f"Final running average loss: {trainer.running_loss:.4f}")

    with lock:
        if training_active_flag[0]:
            shared_logs.append(f"Completed all {max_iters} iterations!")
            trainer.save_checkpoint(trainer.max_iters, is_final=True)
            shared_logs.append("Fine-tuning complete!")
            shared_logs.append(
                f"Final running average loss: {trainer.running_loss:.4f}")
        training_active_flag[0] = False
        progress_data["iter"] = max_iters - 1
        progress_data["progress"] = 1.0
        shared_logs.append(
            f"Final progress: {progress_data['progress']*100:.1f}%")

