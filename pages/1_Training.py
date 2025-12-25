"""Training page for transformer models."""

import streamlit as st
import torch
import os
import threading
import time
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd

from config import ModelConfig
from training_args import TransformerTrainingArgs
from trainer import TransformerTrainer
from dataset import TransformerDataset
from model import TransformerModelWithEinops, TransformerModelWithoutEinops


def train_model_thread(trainer, shared_loss_data, shared_logs, training_active_flag, lock, progress_data):
    """Training thread that updates shared data structures (thread-safe)."""
    try:
        from tqdm import tqdm

        max_iters = trainer.max_iters
        eval_interval = trainer.eval_interval
        print_interval = getattr(trainer, 'print_interval', 100)
        update_interval = 10  # Update progress every 10 iterations for smoother UI

        # Initial log
        print("\nStarting training...")
        print(
            f"Training for {trainer.args.epochs} epochs, {max_iters} total iterations")
        print(
            f"Batch size: {trainer.args.batch_size}, Learning rate: {trainer.args.lr}")
        print(f"Weight decay: {trainer.args.weight_decay}")
        print(f"Evaluating every {eval_interval} iterations\n")

        with lock:
            shared_logs.append("Starting training...")
            shared_logs.append(
                f"Training for {trainer.args.epochs} epochs, {max_iters} total iterations"
            )
            shared_logs.append(
                f"Batch size: {trainer.args.batch_size}, Learning rate: {trainer.args.lr}"
            )
            shared_logs.append(f"Weight decay: {trainer.args.weight_decay}")
            shared_logs.append(
                f"Evaluating every {eval_interval} iterations\n")

        # Create tqdm progress bar for console output
        pbar = tqdm(range(max_iters), desc="Training")

        # Initialize running_loss with None to set it on first iteration
        first_loss_set = False

        for iter_num in pbar:
            # Check if training should stop (thread-safe check)
            with lock:
                if not training_active_flag[0]:
                    shared_logs.append("Training stopped by user.")
                    break

            # Training step
            idx = torch.randint(0, len(trainer.X_train),
                                (trainer.args.batch_size,))
            x_batch = trainer.X_train[idx].to(trainer.device)
            y_batch = trainer.Y_train[idx].to(trainer.device)

            logits = trainer.model(x_batch)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y_batch.view(-1)
            )

            trainer.optimizer.zero_grad()
            loss.backward()
            trainer.optimizer.step()

            # Initialize running_loss with first loss value for better starting point
            if not first_loss_set:
                trainer.running_loss = loss.item()
                first_loss_set = True
            else:
                trainer.running_loss = (
                    trainer.loss_alpha * trainer.running_loss
                    + (1 - trainer.loss_alpha) * loss.item()
                )

            # Update tqdm progress bar (console output)
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{trainer.running_loss:.4f}",
            })

            # Update progress data frequently (every update_interval iterations)
            # Always update on last iteration to ensure 100% progress
            should_update_progress = (iter_num % update_interval == 0 or
                                      iter_num == max_iters - 1 or
                                      iter_num == 0)
            if should_update_progress:
                with lock:
                    progress_data["iter"] = iter_num
                    progress_data["loss"] = loss.item()
                    progress_data["running_loss"] = trainer.running_loss
                    # Calculate progress: (iter_num + 1) / max_iters ensures we reach 100% on last iteration
                    progress_data["progress"] = min(
                        (iter_num + 1) / max_iters, 1.0)
                    # Track all losses for comprehensive graph
                    if "all_losses" in progress_data:
                        progress_data["all_losses"]["iterations"].append(
                            iter_num)
                        progress_data["all_losses"]["current_losses"].append(
                            loss.item())
                        progress_data["all_losses"]["running_losses"].append(
                            trainer.running_loss)

            # Print detailed loss periodically (like original trainer)
            if iter_num % print_interval == 0 and iter_num > 0:
                print(
                    f"\n[Iter {iter_num}] Current loss: {loss.item():.4f}, "
                    f"Running avg: {trainer.running_loss:.4f}"
                )
                with lock:
                    shared_logs.append(
                        f"[Iter {iter_num}] Current loss: {loss.item():.4f}, "
                        f"Running avg: {trainer.running_loss:.4f}"
                    )

            # Evaluate at intervals
            if (iter_num > 0 and iter_num % eval_interval == 0) or iter_num == max_iters - 1:
                losses = trainer.estimate_loss()
                print(
                    f"\n[Iter {iter_num}] Train loss: {losses['train']:.4f}, "
                    f"Val loss: {losses['val']:.4f}"
                )
                # Update tqdm with eval metrics
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "avg_loss": f"{trainer.running_loss:.4f}",
                    "val_loss": f"{losses['val']:.4f}",
                })
                # Thread-safe update
                with lock:
                    shared_loss_data["iterations"].append(iter_num)
                    shared_loss_data["train_losses"].append(losses['train'])
                    shared_loss_data["val_losses"].append(losses['val'])
                    shared_logs.append(
                        f"[Iter {iter_num}] Train loss: {losses['train']:.4f}, "
                        f"Val loss: {losses['val']:.4f}"
                    )
                    # Also update progress with val loss
                    progress_data["val_loss"] = losses['val']

            # Save checkpoint
            if (hasattr(trainer.args, "save_interval") and
                    iter_num % trainer.args.save_interval == 0 and iter_num > 0):
                trainer.save_checkpoint(iter_num)
                print(f"Checkpoint saved at iteration {iter_num}")
                with lock:
                    shared_logs.append(
                        f"Checkpoint saved at iteration {iter_num}")

        # Close tqdm progress bar
        pbar.close()

        # Final save - loop completed normally
        print("\nTraining complete!")
        print(f"Final running average loss: {trainer.running_loss:.4f}")

        with lock:
            if training_active_flag[0]:
                # Training completed all iterations
                shared_logs.append(f"Completed all {max_iters} iterations!")
                trainer.save_checkpoint(trainer.max_iters, is_final=True)
                trainer.save_loss_graph()
                shared_logs.append("Training complete!")
                shared_logs.append(
                    f"Final running average loss: {trainer.running_loss:.4f}")
            training_active_flag[0] = False
            # Ensure progress is exactly 100%
            progress_data["iter"] = max_iters - 1
            progress_data["progress"] = 1.0
            shared_logs.append(
                f"Final progress: {progress_data['progress']*100:.1f}%")
    except Exception as e:
        import traceback
        with lock:
            shared_logs.append(f"Error during training: {str(e)}")
            shared_logs.append(traceback.format_exc())
            training_active_flag[0] = False


st.title("🚂 Training")

# File upload
st.header("1. Upload Training Data")
uploaded_file = st.file_uploader(
    "Upload a text file for training",
    type=["txt"],
    help="Upload a text file to train the model on"
)

# Or use default file
use_default = st.checkbox("Use default training.txt file", value=True)

if not uploaded_file and not use_default:
    st.warning("Please upload a text file or use the default file.")
    st.stop()

# Architecture selection
st.header("2. Model Architecture")
col1, col2 = st.columns(2)

with col1:
    architecture = st.selectbox(
        "Architecture",
        ["GPT", "LLAMA", "OLMO"],
        help="Choose the transformer architecture"
    )

with col2:
    model_size = st.selectbox(
        "Model Size",
        ["small", "full"],
        help="Small for faster training, full for GPT-2/LLaMA size"
    )

use_einops = st.checkbox("Use einops (recommended)", value=True)

# Tokenizer selection
st.header("3. Tokenizer")
tokenizer_type = st.selectbox(
    "Tokenizer Type",
    ["character", "bpe", "sentencepiece"],
    help="Character: simple but large vocab. BPE: subword units. SentencePiece: multilingual support."
)

# Hyperparameters
st.header("4. Training Hyperparameters")
with st.expander("Advanced Settings", expanded=False):
    col1, col2 = st.columns(2)

    with col1:
        batch_size = st.number_input(
            "Batch Size", min_value=1, max_value=128, value=32)
        learning_rate = st.number_input(
            "Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f")
        weight_decay = st.number_input(
            "Weight Decay", min_value=0.0, max_value=1.0, value=1e-2, format="%.5f")

    with col2:
        epochs = st.number_input(
            "Epochs", min_value=1, max_value=100, value=10)
        max_steps_per_epoch = st.number_input(
            "Max Steps per Epoch", min_value=100, max_value=10000, value=500)
        eval_interval = st.number_input(
            "Evaluation Interval", min_value=100, max_value=5000, value=500)
        save_interval = st.number_input(
            "Save Interval", min_value=100, max_value=5000, value=1000)

# Start training button
st.header("5. Start Training")

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    start_training = st.button(
        "🚀 Start Training", type="primary", width='stretch')

with col2:
    stop_training = st.button("⏹️ Stop Training", width='stretch')

if stop_training and st.session_state.training_active:
    with st.session_state.training_lock:
        if "training_active_flag" in st.session_state:
            st.session_state.training_active_flag[0] = False
    st.session_state.training_active = False
    st.rerun()

# Training status and visualization
if start_training:
    if st.session_state.training_active:
        st.warning("Training is already in progress!")
    else:
        # Load text
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")
            st.info(f"Loaded {len(text)} characters from uploaded file")
        else:
            with open("training.txt", "r", encoding="utf-8") as f:
                text = f.read()
            st.info(f"Loaded {len(text)} characters from training.txt")

        # Initialize config
        if architecture == "LLAMA":
            cfg = ModelConfig.llama_small() if model_size == "small" else ModelConfig.llama_full()
        elif architecture == "OLMO":
            cfg = ModelConfig.olmo_small() if model_size == "small" else ModelConfig.olmo_full()
        else:  # GPT
            cfg = ModelConfig.gpt_small() if model_size == "small" else ModelConfig.gpt_full()

        # Create dataset
        dataset = TransformerDataset(
            text, cfg, tokenizer_type=tokenizer_type)
        cfg = dataset.cfg  # Updated with vocab size

        X_train, Y_train = dataset.get_train_data()
        X_val, Y_val = dataset.get_val_data()

        # Initialize model
        device = st.session_state.get_device()
        if use_einops:
            model = TransformerModelWithEinops(cfg)
        else:
            model = TransformerModelWithoutEinops(cfg)
        model = model.to(device)

        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        st.success(
            f"Model initialized: {param_count:.2f}M parameters on {device}")

        # Training args
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        checkpoint_dir = os.path.join("checkpoints", timestamp)
        os.makedirs(checkpoint_dir, exist_ok=True)

        training_args = TransformerTrainingArgs(
            batch_size=batch_size,
            epochs=epochs,
            max_steps_per_epoch=max_steps_per_epoch,
            lr=learning_rate,
            weight_decay=weight_decay,
            save_dir=checkpoint_dir,
            save_interval=save_interval,
            eval_iters=50 if model_size == "small" else 200
        )

        # Create trainer
        trainer = TransformerTrainer(
            model=model,
            args=training_args,
            X_train=X_train,
            Y_train=Y_train,
            X_val=X_val,
            Y_val=Y_val,
            device=device,
            eval_interval=eval_interval,
            tokenizer_type=tokenizer_type
        )

        # Reset loss data (use shared thread-safe structures)
        st.session_state.shared_loss_data = {
            "iterations": [], "train_losses": [], "val_losses": []}
        st.session_state.shared_training_logs.clear()
        st.session_state.training_active = True
        st.session_state.trainer = trainer

        # Create a mutable flag for thread-safe access
        training_active_flag = [True]

        # Progress data for real-time updates
        progress_data = {
            "iter": 0,
            "loss": 0.0,
            "running_loss": 0.0,
            "val_loss": None,
            "progress": 0.0,
            "all_losses": {
                "iterations": [],
                "current_losses": [],
                "running_losses": []
            }
        }

        # Start training thread with thread-safe data structures
        thread = threading.Thread(
            target=train_model_thread,
            args=(
                trainer,
                st.session_state.shared_loss_data,
                st.session_state.shared_training_logs,
                training_active_flag,
                st.session_state.training_lock,
                progress_data
            ),
            daemon=True
        )
        thread.start()
        st.session_state.training_thread = thread
        st.session_state.training_active_flag = training_active_flag
        st.session_state.progress_data = progress_data

        st.success("Training started! Check the visualization below.")
        time.sleep(0.5)  # Give thread a moment to start
        st.rerun()

# Display training status
# Check if thread is still alive and training flag status
if st.session_state.training_thread is not None:
    thread_alive = st.session_state.training_thread.is_alive()
    # Check training_active_flag to see actual training status
    training_flag_active = True
    if "training_active_flag" in st.session_state:
        with st.session_state.training_lock:
            training_flag_active = st.session_state.training_active_flag[0]

    # Only mark as completed if thread is dead AND flag is False (normal completion)
    # OR if we have explicit completion logs
    if not thread_alive and st.session_state.training_active:
        # Check logs to determine why thread stopped
        if st.session_state.shared_training_logs:
            # Check last 3 logs
            last_logs = list(st.session_state.shared_training_logs)[-3:]
            last_logs_str = " ".join(last_logs)
            if "Training complete!" in last_logs_str or "Completed all" in last_logs_str:
                st.session_state.training_active = False
                st.success("✅ Training completed!")
            elif "Error during training" in last_logs_str:
                st.session_state.training_active = False
                st.error("❌ Training error occurred. Check logs for details.")
            elif "Training stopped by user" in last_logs_str:
                st.session_state.training_active = False
                st.info("⏹️ Training stopped by user.")
            elif not training_flag_active:
                # Flag is False but no explicit message - thread completed normally
                st.session_state.training_active = False
                st.success("✅ Training completed!")
            # else: thread died but flag still True - might be a race condition, don't mark as done
        elif not training_flag_active:
            # No logs but flag is False - assume normal completion
            st.session_state.training_active = False
            st.success("✅ Training completed!")
        # else: thread dead but flag True and no logs - might be starting up, don't mark as done

if st.session_state.training_active:
    # Real-time progress bar and metrics
    if "progress_data" in st.session_state:
        progress_data = st.session_state.progress_data
        with st.session_state.training_lock:
            # Thread-safe read
            current_iter = progress_data.get("iter", 0)
            current_loss = progress_data.get("loss", 0.0)
            running_loss = progress_data.get("running_loss", 0.0)
            val_loss = progress_data.get("val_loss")
            progress = progress_data.get("progress", 0.0)

        st.header("📊 Training Progress")

        # Progress bar
        st.progress(
            progress, text=f"Iteration {current_iter} / {st.session_state.trainer.max_iters if st.session_state.trainer else '?'}")

        # Real-time metrics
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Current Loss", f"{current_loss:.4f}")
        with metric_cols[1]:
            st.metric("Running Avg", f"{running_loss:.4f}")
        with metric_cols[2]:
            if val_loss is not None:
                st.metric("Val Loss", f"{val_loss:.4f}")
            else:
                st.metric("Val Loss", "Pending...")
        with metric_cols[3]:
            st.metric("Progress", f"{progress*100:.1f}%")
    else:
        st.header("📊 Training Progress")

    # Copy from shared data to display (thread-safe read)
    with st.session_state.training_lock:
        loss_data = {
            "iterations": list(st.session_state.shared_loss_data["iterations"]),
            "train_losses": list(st.session_state.shared_loss_data["train_losses"]),
            "val_losses": list(st.session_state.shared_loss_data["val_losses"])
        }
        training_logs = list(st.session_state.shared_training_logs)
        # Get all losses data if available
        all_losses_data = None
        if "progress_data" in st.session_state and "all_losses" in st.session_state.progress_data:
            all_losses_data = {
                "iterations": list(st.session_state.progress_data["all_losses"]["iterations"]),
                "current_losses": list(st.session_state.progress_data["all_losses"]["current_losses"]),
                "running_losses": list(st.session_state.progress_data["all_losses"]["running_losses"])
            }

    # Also update session state for persistence
    st.session_state.loss_data = loss_data
    st.session_state.training_logs = training_logs
    if all_losses_data:
        st.session_state.all_losses_data = all_losses_data

    # Graph 1: All losses (current and running average) - updates every 10 iterations
    if all_losses_data and len(all_losses_data["iterations"]) > 0:
        st.subheader("📈 All Losses (Real-time)")
        df_all = pd.DataFrame({
            "Iteration": all_losses_data["iterations"],
            "Current Loss": all_losses_data["current_losses"],
            "Running Avg Loss": all_losses_data["running_losses"]
        })

        fig_all = go.Figure()
        fig_all.add_trace(go.Scatter(
            x=df_all["Iteration"],
            y=df_all["Current Loss"],
            mode="lines",
            name="Current Loss",
            line={"color": "orange", "width": 1},
            opacity=0.7
        ))
        fig_all.add_trace(go.Scatter(
            x=df_all["Iteration"],
            y=df_all["Running Avg Loss"],
            mode="lines",
            name="Running Avg Loss",
            line={"color": "purple", "width": 2}
        ))
        fig_all.update_layout(
            title="All Training Losses (updated every 10 iterations)",
            xaxis_title="Iteration",
            yaxis_title="Loss",
            hovermode="x unified",
            height=400,
            yaxis={"range": [0, None]}  # Always start y-axis at 0
        )
        st.plotly_chart(fig_all, width='stretch')

    # Graph 2: Evaluation losses (train/val) - updates at evaluation intervals
    if loss_data["iterations"]:
        st.subheader("📊 Evaluation Losses (Train/Val)")
        df = pd.DataFrame({
            "Iteration": loss_data["iterations"],
            "Train Loss": loss_data["train_losses"],
            "Val Loss": loss_data["val_losses"]
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Iteration"],
            y=df["Train Loss"],
            mode="lines+markers",
            name="Train Loss",
            line={"color": "blue"}
        ))
        fig.add_trace(go.Scatter(
            x=df["Iteration"],
            y=df["Val Loss"],
            mode="lines+markers",
            name="Val Loss",
            line={"color": "red"}
        ))
        fig.update_layout(
            title="Training and Validation Loss (evaluated every 500 iterations)",
            xaxis_title="Iteration",
            yaxis_title="Loss",
            hovermode="x unified",
            height=400
        )
        st.plotly_chart(fig, width='stretch')

        # Auto-refresh note
        st.caption("💡 Page auto-refreshes every 2 seconds while training.")

        # Auto-refresh while training
        if st.session_state.training_active:
            time.sleep(2)
            st.rerun()
    else:
        if st.session_state.training_active:
            st.info("⏳ Waiting for first evaluation (at the 500th iteration).")
            # Auto-refresh while waiting
            time.sleep(2)
            st.rerun()

    # Training logs (like console output)
    if training_logs:
        st.header("📝 Training Logs (Console Output)")
        # Show logs in a scrollable container
        with st.expander("View All Logs", expanded=True):
            # Show all logs (they're already limited by deque maxlen=200)
            log_text = "\n".join(training_logs)
            st.text_area(
                "Logs",
                value=log_text,
                height=400,
                label_visibility="collapsed",
                disabled=True
            )
        st.caption(f"Showing {len(training_logs)} log entries")
else:
    if st.session_state.loss_data["iterations"]:
        st.header("📊 Final Training Results")
        df = pd.DataFrame({
            "Iteration": st.session_state.loss_data["iterations"],
            "Train Loss": st.session_state.loss_data["train_losses"],
            "Val Loss": st.session_state.loss_data["val_losses"]
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Iteration"],
            y=df["Train Loss"],
            mode="lines+markers",
            name="Train Loss",
            line={"color": "blue"}
        ))
        fig.add_trace(go.Scatter(
            x=df["Iteration"],
            y=df["Val Loss"],
            mode="lines+markers",
            name="Val Loss",
            line={"color": "red"}
        ))
        fig.update_layout(
            title="Training and Validation Loss",
            xaxis_title="Iteration",
            yaxis_title="Loss",
            hovermode="x unified"
        )
        st.plotly_chart(fig, width='stretch')
