"""Fine-tuning page for supervised fine-tuning (SFT)."""

import os
import threading
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pretraining.model.model_loader import load_model_from_checkpoint
from pretraining.tokenization.tokenizer import (
    CharacterTokenizer,
    SimpleBPETokenizer,
    BPETokenizer,
    SentencePieceTokenizer,
)
from finetuning.data.sft_dataset import SFTDataset
from finetuning.training.sft_trainer import SFTTrainer
from finetuning.training.finetuning_args import FinetuningArgs
from finetuning.training.sft_training_ui import train_sft_model_thread
from pretraining.training.training_ui import initialize_training_state
from ui_components import render_checkpoint_selector


def _start_finetuning_workflow(
    selected_checkpoint_path,
    uploaded_csv,
    batch_size,
    lr,
    weight_decay,
    epochs,
    max_steps_per_epoch,
    eval_interval,
    save_interval,
    max_length,
    use_lora,
    lora_rank,
    lora_alpha,
    lora_dropout,
    lora_target_modules,
):
    """Start the fine-tuning workflow."""
    device = st.session_state.get_device()

    # Load pre-trained model
    with st.spinner("Loading pre-trained model..."):
        model, cfg, checkpoint = st.session_state.load_model_from_checkpoint(
            selected_checkpoint_path, device
        )
        model.train()  # Set to training mode

    # Apply LoRA if selected
    if use_lora:
        with st.spinner("Applying LoRA adapters..."):
            from finetuning.peft.lora_utils import convert_model_to_lora, count_lora_parameters
            model = convert_model_to_lora(
                model,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
                target_modules=lora_target_modules,
            )
            param_counts = count_lora_parameters(model)
            st.success(
                f"✅ LoRA applied! Training {param_counts['lora']:,} LoRA parameters "
                f"({param_counts['lora']/param_counts['total']*100:.2f}% of total). "
                f"{param_counts['frozen']:,} base parameters frozen."
            )

    tokenizer_type = checkpoint.get("tokenizer_type", "character")

    # Create tokenizer (must match pre-trained model)
    if tokenizer_type == "character":
        if os.path.exists("training.txt"):
            with open("training.txt", "r", encoding="utf-8") as f:
                text = f.read()
            tokenizer = CharacterTokenizer(text)
        else:
            st.error("training.txt not found. Cannot load character tokenizer.")
            st.stop()
    elif tokenizer_type == "bpe-simple":
        if os.path.exists("training.txt"):
            with open("training.txt", "r", encoding="utf-8") as f:
                text = f.read()
            vocab_size = cfg.d_vocab if hasattr(cfg, 'd_vocab') else 1000
            tokenizer = SimpleBPETokenizer(text, vocab_size=vocab_size)
        else:
            st.error(
                "training.txt not found. Cannot recreate Simple BPE tokenizer.")
            st.stop()
    elif tokenizer_type == "bpe-tiktoken" or tokenizer_type == "bpe":
        tokenizer = BPETokenizer()
    elif tokenizer_type == "sentencepiece":
        if os.path.exists("training.txt"):
            with open("training.txt", "r", encoding="utf-8") as f:
                text = f.read()
            vocab_size = cfg.d_vocab if hasattr(cfg, 'd_vocab') else 10000
            tokenizer = SentencePieceTokenizer(text, vocab_size=vocab_size)
        else:
            st.error(
                "training.txt not found. Cannot recreate SentencePiece tokenizer.")
            st.stop()
    else:
        st.error(f"Tokenizer type {tokenizer_type} not supported.")
        st.stop()

    # Save CSV temporarily (use default if no upload)
    csv_path = "temp_sft_data.csv"
    if uploaded_csv:
        with open(csv_path, "wb") as f:
            f.write(uploaded_csv.getbuffer())
        df = pd.read_csv(csv_path)
        st.info(f"Loaded CSV with {len(df)} rows.")
    elif os.path.exists("finetuning.csv"):
        # Use default file
        import shutil
        shutil.copy("finetuning.csv", csv_path)
        df = pd.read_csv(csv_path)
        st.info(f"Using default finetuning.csv with {len(df)} rows.")
    else:
        st.error("No CSV file provided and finetuning.csv not found.")
        st.stop()

    # Create SFT dataset
    with st.spinner("Creating fine-tuning dataset..."):
        dataset = SFTDataset(
            csv_path=csv_path,
            tokenizer=tokenizer,
            max_length=max_length,
        )
        dataset.print_info()

    X_train, Y_train, masks_train = dataset.get_train_data()
    X_val, Y_val, masks_val = dataset.get_val_data()

    # Create save directory: checkpoints/{timestamp}/sft/
    checkpoint_dir = Path(selected_checkpoint_path).parent
    timestamp = checkpoint_dir.name
    sft_dir = checkpoint_dir / "sft"
    sft_dir.mkdir(exist_ok=True)

    # Training args
    training_args = FinetuningArgs(
        batch_size=batch_size,
        epochs=epochs,
        max_steps_per_epoch=max_steps_per_epoch,
        lr=lr,
        weight_decay=weight_decay,
        save_dir=str(sft_dir),
        save_interval=save_interval,
        eval_iters=50,
        use_lora=use_lora,
        lora_rank=lora_rank if use_lora else 8,
        lora_alpha=lora_alpha if use_lora else 8.0,
        lora_dropout=lora_dropout if use_lora else 0.0,
        lora_target_modules=lora_target_modules if use_lora else "all",
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        X_train=X_train,
        Y_train=Y_train,
        masks_train=masks_train,
        X_val=X_val,
        Y_val=Y_val,
        masks_val=masks_val,
        device=device,
        eval_interval=eval_interval,
        tokenizer_type=tokenizer_type,
    )

    # Initialize training state
    st.session_state.shared_loss_data = {
        "iterations": [], "train_losses": [], "val_losses": []
    }
    st.session_state.shared_training_logs.clear()
    st.session_state.training_active = True
    st.session_state.trainer = trainer

    training_active_flag = [True]
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

    # Start training thread
    thread = threading.Thread(
        target=train_sft_model_thread,
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

    st.success("Fine-tuning started! Check the visualization below.")
    time.sleep(0.5)
    st.rerun()


def _render_all_losses_graph(all_losses_data):
    """Render all losses graph."""
    st.subheader("📈 All Losses (Real-time)")
    df_all = pd.DataFrame({
        "Iteration": all_losses_data["iterations"],
        "Current Loss": all_losses_data["current_losses"],
        "Running Avg Loss": all_losses_data["running_losses"]
    })

    fig_all = go.Figure()
    fig_all.add_trace(go.Scatter(
        x=df_all["Iteration"], y=df_all["Current Loss"],
        mode="lines", name="Current Loss",
        line={"color": "orange", "width": 1}, opacity=0.7
    ))
    fig_all.add_trace(go.Scatter(
        x=df_all["Iteration"], y=df_all["Running Avg Loss"],
        mode="lines", name="Running Avg Loss",
        line={"color": "purple", "width": 2}
    ))
    fig_all.update_layout(
        title="All Fine-Tuning Losses (updated every 10 iterations)",
        xaxis_title="Iteration", yaxis_title="Loss",
        hovermode="x unified", height=400,
        yaxis={"range": [0, None]}
    )
    st.plotly_chart(fig_all, width='stretch')


def _render_eval_losses_graph(loss_data):
    """Render evaluation losses graph."""
    st.subheader("📊 Evaluation Losses (Train/Val)")
    df = pd.DataFrame({
        "Iteration": loss_data["iterations"],
        "Train Loss": loss_data["train_losses"],
        "Val Loss": loss_data["val_losses"]
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Iteration"], y=df["Train Loss"],
        mode="lines+markers", name="Train Loss",
        line={"color": "blue"}
    ))
    fig.add_trace(go.Scatter(
        x=df["Iteration"], y=df["Val Loss"],
        mode="lines+markers", name="Val Loss",
        line={"color": "red"}
    ))
    fig.update_layout(
        title="Training and Validation Loss (evaluated every 500 iterations)",
        xaxis_title="Iteration", yaxis_title="Loss",
        hovermode="x unified", height=400
    )
    st.plotly_chart(fig, width='stretch')


def _render_active_training_ui():
    """Render UI for active training."""
    if "progress_data" in st.session_state:
        progress_data = st.session_state.progress_data
        with st.session_state.training_lock:
            current_iter = progress_data.get("iter", 0)
            current_loss = progress_data.get("loss", 0.0)
            running_loss = progress_data.get("running_loss", 0.0)
            val_loss = progress_data.get("val_loss")
            progress = progress_data.get("progress", 0.0)

        st.header("📊 Fine-Tuning Progress")
        st.progress(
            progress, text=f"Iteration {current_iter} / {st.session_state.trainer.max_iters if st.session_state.trainer else '?'}")

        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.metric("Current Loss", f"{current_loss:.4f}")
        with metric_cols[1]:
            st.metric("Running Avg", f"{running_loss:.4f}")
        with metric_cols[2]:
            st.metric(
                "Val Loss", f"{val_loss:.4f}" if val_loss is not None else "Pending...")
        with metric_cols[3]:
            st.metric("Progress", f"{progress*100:.1f}%")

    # Get loss data (thread-safe)
    with st.session_state.training_lock:
        loss_data = {
            "iterations": list(st.session_state.shared_loss_data["iterations"]),
            "train_losses": list(st.session_state.shared_loss_data["train_losses"]),
            "val_losses": list(st.session_state.shared_loss_data["val_losses"])
        }
        training_logs = list(st.session_state.shared_training_logs)
        all_losses_data = None
        if "progress_data" in st.session_state and "all_losses" in st.session_state.progress_data:
            all_losses_data = {
                "iterations": list(st.session_state.progress_data["all_losses"]["iterations"]),
                "current_losses": list(st.session_state.progress_data["all_losses"]["current_losses"]),
                "running_losses": list(st.session_state.progress_data["all_losses"]["running_losses"])
            }

    st.session_state.loss_data = loss_data
    st.session_state.training_logs = training_logs
    if all_losses_data:
        st.session_state.all_losses_data = all_losses_data

    # Render graphs
    if all_losses_data and len(all_losses_data["iterations"]) > 0:
        _render_all_losses_graph(all_losses_data)

    if loss_data["iterations"]:
        _render_eval_losses_graph(loss_data)
        st.caption("💡 Page auto-refreshes every 2 seconds while fine-tuning.")
        if st.session_state.training_active:
            time.sleep(2)
            st.rerun()
    else:
        if st.session_state.training_active:
            st.info("⏳ Waiting for first evaluation (at the 500th iteration).")
            time.sleep(2)
            st.rerun()

    # Training logs
    if training_logs:
        st.header("📝 Fine-Tuning Logs (Console Output)")
        with st.expander("View All Logs", expanded=True):
            log_text = "\n".join(training_logs)
            st.text_area("Logs", value=log_text, height=400,
                         label_visibility="collapsed", disabled=True)
        st.caption(f"Showing {len(training_logs)} log entries")


def _render_completed_training_ui():
    """Render UI for completed training."""
    if st.session_state.loss_data["iterations"]:
        st.header("📊 Final Fine-Tuning Results")
        df = pd.DataFrame({
            "Iteration": st.session_state.loss_data["iterations"],
            "Train Loss": st.session_state.loss_data["train_losses"],
            "Val Loss": st.session_state.loss_data["val_losses"]
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Iteration"], y=df["Train Loss"],
            mode="lines+markers", name="Train Loss",
            line={"color": "blue"}
        ))
        fig.add_trace(go.Scatter(
            x=df["Iteration"], y=df["Val Loss"],
            mode="lines+markers", name="Val Loss",
            line={"color": "red"}
        ))
        fig.update_layout(
            title="Training and Validation Loss",
            xaxis_title="Iteration", yaxis_title="Loss",
            hovermode="x unified"
        )
        st.plotly_chart(fig, width='stretch')


def _handle_training_completion(training_flag_active: bool):
    """Handle training completion logic."""
    if st.session_state.shared_training_logs:
        last_logs = list(st.session_state.shared_training_logs)[-3:]
        last_logs_str = " ".join(last_logs)
        if "Fine-tuning complete!" in last_logs_str or "Completed all" in last_logs_str:
            st.session_state.training_active = False
            st.success("✅ Fine-tuning completed!")
        elif "Error during fine-tuning" in last_logs_str:
            st.session_state.training_active = False
            st.error("❌ Fine-tuning error occurred. Check logs for details.")
        elif "Fine-tuning stopped by user" in last_logs_str:
            st.session_state.training_active = False
            st.info("⏹️ Fine-tuning stopped by user.")
        elif not training_flag_active:
            st.session_state.training_active = False
            st.success("✅ Fine-tuning completed!")
    elif not training_flag_active:
        st.session_state.training_active = False
        st.success("✅ Fine-tuning completed!")


def _display_training_status():
    """Display training status and visualizations."""
    # Check training status
    if st.session_state.training_thread is not None:
        thread_alive = st.session_state.training_thread.is_alive()
        training_flag_active = True
        if "training_active_flag" in st.session_state:
            with st.session_state.training_lock:
                training_flag_active = st.session_state.training_active_flag[0]

        if not thread_alive and st.session_state.training_active:
            _handle_training_completion(training_flag_active)

    if st.session_state.training_active:
        _render_active_training_ui()
    else:
        _render_completed_training_ui()


st.title("🎯 Fine-Tuning (SFT)")

# Initialize training state
initialize_training_state()

# Checkpoint selection
selected_checkpoint = render_checkpoint_selector(
    header="1. Select Pre-Trained Checkpoint",
    filter_finetuned=True,
    help_text="Select a pre-trained model to fine-tune",
    no_checkpoints_message="No pre-trained checkpoints found. Please pre-train a model first."
)

# Fine-tuning method selection
st.header("2. Fine-Tuning Method")
fine_tuning_method = st.radio(
    "Select fine-tuning method",
    ["Full Parameter Fine-Tuning", "LoRA (Parameter-Efficient)"],
    help="Full Parameter: Updates all model weights. LoRA: Only trains small adapter matrices (faster, less memory)."
)

# LoRA options (only show if LoRA selected)
use_lora = fine_tuning_method == "LoRA (Parameter-Efficient)"
lora_rank = 8
lora_alpha = 8.0
lora_dropout = 0.0
lora_target_modules = "all"

if use_lora:
    with st.expander("LoRA Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            lora_rank = st.number_input(
                "LoRA Rank (r)",
                min_value=1,
                max_value=128,
                value=8,
                help="Dimension of low-rank matrices. Higher = more parameters, more capacity. Typical: 4-16"
            )
            lora_alpha = st.number_input(
                "LoRA Alpha (α)",
                min_value=1.0,
                max_value=256.0,
                value=8.0,
                help="Scaling factor. Typically set equal to rank. Higher = stronger LoRA influence."
            )
        with col2:
            lora_dropout = st.number_input(
                "LoRA Dropout",
                min_value=0.0,
                max_value=0.5,
                value=0.0,
                step=0.05,
                format="%.2f",
                help="Dropout rate for LoRA adapters. Helps prevent overfitting."
            )
            lora_target_modules = st.selectbox(
                "Target Modules",
                ["all", "attention", "mlp"],
                help="Which layers to apply LoRA to. 'all' applies to both attention and MLP layers."
            )

        st.info(
            f"💡 LoRA will train ~{lora_rank * 2} parameters per weight matrix "
            f"(rank={lora_rank}). This is much smaller than full fine-tuning!"
        )

# CSV upload
st.header("3. Upload Training Data")
uploaded_csv = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="CSV should have two columns: 'prompt'/'response' or 'instruction'/'output'. Each row is a training example. You can format your data however you like (including with instruction templates already applied). If no file is uploaded, the default finetuning.csv will be used."
)

# Use default file if no upload
if uploaded_csv is None:
    if os.path.exists("finetuning.csv"):
        st.info(
            "📄 Using default finetuning.csv file. Upload a different file to override.")
        # Preview default CSV
        df = pd.read_csv("finetuning.csv")
        st.dataframe(df.head(), use_container_width=True)
        st.caption(f"Total rows: {len(df)}")
    else:
        st.warning(
            "No CSV file uploaded and finetuning.csv not found. Please upload a CSV file.")
else:
    # Preview uploaded CSV
    df = pd.read_csv(uploaded_csv)
    st.dataframe(df.head(), use_container_width=True)
    st.caption(f"Total rows: {len(df)}")

# Hyperparameters
st.header("4. Fine-Tuning Hyperparameters")
col1, col2 = st.columns(2)
with col1:
    batch_size = st.number_input(
        "Batch Size", min_value=1, max_value=32, value=4)
    learning_rate = st.number_input(
        "Learning Rate", min_value=1e-6, max_value=1e-3, value=1e-5, format="%.6f"
    )
    weight_decay = st.number_input(
        "Weight Decay", min_value=0.0, max_value=1.0, value=0.01, format="%.5f"
    )
with col2:
    epochs = st.number_input("Epochs", min_value=1, max_value=10, value=3)
    max_steps_per_epoch = st.number_input(
        "Max Steps per Epoch", min_value=100, max_value=5000, value=1000
    )
    eval_interval = st.number_input(
        "Evaluation Interval", min_value=100, max_value=2000, value=500
    )
    save_interval = st.number_input(
        "Save Interval", min_value=100, max_value=2000, value=500
    )

max_length = st.number_input(
    "Max Sequence Length", min_value=128, max_value=2048, value=512,
    help="Maximum length for prompt+response sequences"
)

# Start fine-tuning button
st.header("5. Start Fine-Tuning")
col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    start_finetuning = st.button("🚀 Start Fine-Tuning", type="primary")
with col2:
    stop_training = st.button("⏹️ Stop Fine-Tuning")

if stop_training and st.session_state.training_active:
    with st.session_state.training_lock:
        if "training_active_flag" in st.session_state:
            st.session_state.training_active_flag[0] = False
    st.session_state.training_active = False
    st.rerun()

if start_finetuning:
    if not uploaded_csv and not os.path.exists("finetuning.csv"):
        st.error("Please upload a CSV file or ensure finetuning.csv exists.")
    elif st.session_state.training_active:
        st.warning("Training is already in progress!")
    else:
        _start_finetuning_workflow(
            selected_checkpoint["path"],
            uploaded_csv,
            batch_size,
            learning_rate,
            weight_decay,
            epochs,
            max_steps_per_epoch,
            eval_interval,
            save_interval,
            max_length,
            use_lora,
            lora_rank,
            lora_alpha,
            lora_dropout,
            lora_target_modules,
        )

# Display training status
_display_training_status()
