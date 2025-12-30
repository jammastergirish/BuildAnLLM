"""Shared utilities for training threads."""

import threading
import time
import traceback
from collections import deque
from typing import Dict, Any, List

from tqdm import tqdm


def run_training_thread(
    trainer,
    shared_loss_data: Dict[str, List],
    shared_logs: deque,
    training_active_flag: List[bool],
    lock: threading.Lock,
    progress_data: Dict[str, Any]
) -> None:
    """
    Generic training thread that drives the training loop for any Trainer.
    
    The trainer object must implement:
    - max_iters (int)
    - eval_interval (int)
    - args (object with epochs, batch_size, lr, weight_decay, save_interval)
    - running_loss (float)
    - train_single_step() -> dict (must return 'loss', 'running_loss')
    - estimate_loss() -> dict ('train', 'val')
    - save_checkpoint(iter_num)
    - save_loss_graph() (optional)
    """
    try:
        max_iters = trainer.max_iters
        eval_interval = trainer.eval_interval
        print_interval = getattr(trainer, 'print_interval', 100)

        _log_training_start(trainer, shared_logs, lock, eval_interval)

        pbar = tqdm(range(max_iters), desc="Training")
        
        # We don't need 'first_loss_set' logic if the trainer handles running_loss initialization
        # which our standard train_single_step does.

        for iter_num in pbar:
            if not _check_training_active(training_active_flag, lock, shared_logs):
                break

            # Perform Step
            # Calls the trainer's self-contained step method
            metrics = trainer.train_single_step()
            loss = metrics["loss"]
            running_loss = metrics["running_loss"]

            pbar.set_postfix({
                "loss": f"{loss:.4f}",
                "avg_loss": f"{running_loss:.4f}",
            })

            # Update UI Progress Data
            _update_progress(progress_data, iter_num, loss,
                             running_loss, max_iters, lock)

            # Log to Text Console
            if iter_num % print_interval == 0 and iter_num > 0:
                _log_iteration(iter_num, loss, running_loss,
                               shared_logs, lock)

            # Evaluate
            if (iter_num > 0 and iter_num % eval_interval == 0) or iter_num == max_iters - 1:
                _evaluate_and_log(trainer, iter_num, shared_loss_data,
                                  shared_logs, progress_data, lock, pbar)

            # Save Checkpoint
            if (hasattr(trainer.args, "save_interval") and
                    iter_num % trainer.args.save_interval == 0 and iter_num > 0):
                trainer.save_checkpoint(iter_num)
                _log_checkpoint_saved(iter_num, shared_logs, lock)

        pbar.close()
        _finalize_training(trainer, max_iters, training_active_flag,
                           shared_logs, progress_data, lock)

    except Exception as e:
        error_msg = f"Error during training: {str(e)}"
        traceback_str = traceback.format_exc()
        print(error_msg)
        print(traceback_str)
        with lock:
            shared_logs.append("=" * 80)
            shared_logs.append("ERROR DETECTED")
            shared_logs.append("=" * 80)
            shared_logs.append(error_msg)
            shared_logs.append("")
            shared_logs.append("Full traceback:")
            shared_logs.append(traceback_str)
            shared_logs.append("=" * 80)
            training_active_flag[0] = False
            # Set end time for proper timing calculation
            if "training_end_time" not in progress_data:
                progress_data["training_end_time"] = time.time()


# --- Helper Functions ---

def _log_training_start(trainer, shared_logs, lock, eval_interval):
    """Log training start information."""
    print("\nStarting training...")
    print(
        f"Training for {trainer.args.epochs} epochs, {trainer.max_iters} total iterations")
    print(
        f"Batch size: {trainer.args.batch_size}, Learning rate: {trainer.args.lr}")
    print(f"Weight decay: {trainer.args.weight_decay}")
    print(f"Evaluating every {eval_interval} iterations\n")

    with lock:
        shared_logs.extend([
            "Starting training...",
            f"Training for {trainer.args.epochs} epochs, {trainer.max_iters} total iterations",
            f"Batch size: {trainer.args.batch_size}, Learning rate: {trainer.args.lr}",
            f"Weight decay: {trainer.args.weight_decay}",
            f"Evaluating every {eval_interval} iterations\n"
        ])


def _check_training_active(flag, lock, logs):
    """Check if training should continue."""
    with lock:
        if not flag[0]:
            logs.append("Training stopped by user.")
            return False
    return True


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
    print("\nTraining complete!")
    print(f"Final running average loss: {trainer.running_loss:.4f}")

    with lock:
        if training_active_flag[0]:
            shared_logs.append(f"Completed all {max_iters} iterations!")
            trainer.save_checkpoint(trainer.max_iters, is_final=True)
            if hasattr(trainer, "save_loss_graph"):
                trainer.save_loss_graph()
            shared_logs.append("Training complete!")
            shared_logs.append(
                f"Final running average loss: {trainer.running_loss:.4f}")
        training_active_flag[0] = False
        progress_data["iter"] = max_iters - 1
        progress_data["progress"] = 1.0
        shared_logs.append(
            f"Final progress: {progress_data['progress']*100:.1f}%")
