"""Training page for transformer models."""

import streamlit as st
import os
import threading
import time
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd

from config import ModelConfig, Architecture, PositionalEncoding, Normalization, Activation
from pretraining.training.training_args import TransformerTrainingArgs
from pretraining.training.trainer import TransformerTrainer
from pretraining.data.dataset import TransformerDataset
from pretraining.model.model import TransformerModelWithEinops, TransformerModelWithoutEinops
from pretraining.training.training_ui import initialize_training_state, train_model_thread
from ui_components import render_model_config_ui


# Define helper functions first
def _create_model_config(model_config: dict) -> ModelConfig:
    """Create ModelConfig from UI config dict."""
    return ModelConfig(
        architecture=Architecture.GPT,  # Base, doesn't matter
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        n_layers=model_config["n_layers"],
        n_ctx=model_config["n_ctx"],
        d_head=model_config["d_head"],
        d_mlp=model_config["d_mlp"],
        positional_encoding=PositionalEncoding(
            model_config["positional_encoding"]),
        normalization=Normalization(model_config["normalization"]),
        activation=Activation(model_config["activation"]),
        rope_theta=model_config.get("rope_theta", 10000.0),
    )


def _start_training_workflow(uploaded_file, model_config, tokenizer_type, use_einops,
                             batch_size, lr, weight_decay, epochs, max_steps_per_epoch,
                             eval_interval, save_interval):
    """Start the training workflow."""
    # Load text
    if uploaded_file:
        text = uploaded_file.read().decode("utf-8")
    else:
        with open("training.txt", "r", encoding="utf-8") as f:
            text = f.read()
    st.info(f"Loaded {len(text):,} characters.")

    # Create config
    cfg = _create_model_config(model_config)

    # Create dataset
    dataset = TransformerDataset(text, cfg, tokenizer_type=tokenizer_type)
    cfg = dataset.cfg

    X_train, Y_train = dataset.get_train_data()
    X_val, Y_val = dataset.get_val_data()

    # Initialize model
    device = st.session_state.get_device()
    model = TransformerModelWithEinops(
        cfg) if use_einops else TransformerModelWithoutEinops(cfg)
    model = model.to(device)

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    st.success(f"Model initialized: {param_count:.2f}M parameters on {device}")

    # Training args
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    checkpoint_dir = os.path.join("checkpoints", timestamp)
    os.makedirs(checkpoint_dir, exist_ok=True)

    training_args = TransformerTrainingArgs(
        batch_size=batch_size,
        epochs=epochs,
        max_steps_per_epoch=max_steps_per_epoch,
        lr=lr,
        weight_decay=weight_decay,
        save_dir=checkpoint_dir,
        save_interval=save_interval,
        eval_iters=50 if model_config["model_size"] == "small" else 200
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
    time.sleep(0.5)
    st.rerun()


def _handle_training_completion(training_flag_active: bool):
    """Handle training completion logic."""
    if st.session_state.shared_training_logs:
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
            st.session_state.training_active = False
            st.success("✅ Training completed!")
    elif not training_flag_active:
        st.session_state.training_active = False
        st.success("✅ Training completed!")


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
        title="All Training Losses (updated every 10 iterations)",
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

        st.header("📊 Training Progress")
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
        st.caption("💡 Page auto-refreshes every 2 seconds while training.")
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
        st.header("📝 Training Logs (Console Output)")
        with st.expander("View All Logs", expanded=True):
            log_text = "\n".join(training_logs)
            st.text_area("Logs", value=log_text, height=400,
                         label_visibility="collapsed", disabled=True)
        st.caption(f"Showing {len(training_logs)} log entries")


def _render_completed_training_ui():
    """Render UI for completed training."""
    if st.session_state.loss_data["iterations"]:
        st.header("📊 Final Training Results")
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


st.title("🚂 Training")

# Initialize training state
initialize_training_state()

# File upload
st.header("1. Upload Training Data")
uploaded_file = st.file_uploader(
    "Upload a text file for training",
    type=["txt"],
    help="Upload a text file to train the model on. If no file is uploaded, the default training.txt file will be used."
)

# Model configuration UI
st.header("2. Model Architecture")
model_config = render_model_config_ui()

use_einops = st.checkbox("Use einops (recommended)", value=True)

# Tokenizer selection
st.header("3. Tokenizer")
tokenizer_options = ["character", "bpe", "sentencepiece"]
current_tokenizer = model_config.get("tokenizer_type", "bpe")
tokenizer_index = tokenizer_options.index(
    current_tokenizer) if current_tokenizer in tokenizer_options else 1
tokenizer_type = st.selectbox(
    "Tokenizer Type",
    tokenizer_options,
    index=tokenizer_index,
    help="Character: simple but large vocab. BPE: subword units (GPT-2 style). SentencePiece: multilingual support (LLaMA/OLMo style)."
)
model_config["tokenizer_type"] = tokenizer_type

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

# Training logic
if start_training:
    if st.session_state.training_active:
        st.warning("Training is already in progress!")
    else:
        _start_training_workflow(
            uploaded_file, model_config, tokenizer_type, use_einops,
            batch_size, learning_rate, weight_decay, epochs,
            max_steps_per_epoch, eval_interval, save_interval
        )

# Display training status and visualization
_display_training_status()
