"""Pre-training page for transformer models."""

import streamlit as st
import os
import threading
import time
from datetime import datetime

from config import ModelConfig, Architecture, PositionalEncoding, Normalization, Activation
from pretraining.training.training_args import TransformerTrainingArgs
from pretraining.training.trainer import TransformerTrainer
from pretraining.data.dataset import TransformerDataset
from pretraining.model.model import TransformerModelWithEinops, TransformerModelWithoutEinops
from pretraining.training.training_ui import initialize_training_state, train_model_thread
from ui_components import (
    render_model_config_ui, render_model_architecture_diagram, render_model_equations,
    render_model_code_snippets, format_elapsed_time, get_total_training_time,
    render_training_metrics, render_all_losses_graph, render_eval_losses_graph,
    render_completed_training_ui
)


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
    st.session_state.training_start_time = time.time()

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
    # Record end time
    if "training_start_time" in st.session_state and "training_end_time" not in st.session_state:
        st.session_state.training_end_time = time.time()

    total_time = get_total_training_time()

    if st.session_state.shared_training_logs:
        last_logs = list(st.session_state.shared_training_logs)[-3:]
        last_logs_str = " ".join(last_logs)
        if "Training complete!" in last_logs_str or "Completed all" in last_logs_str:
            st.session_state.training_active = False
            st.success(
                f"✅ Training completed! Total time: {format_elapsed_time(total_time)}")
        elif "Error during training" in last_logs_str:
            st.session_state.training_active = False
            st.error("❌ Training error occurred. Check logs for details.")
        elif "Training stopped by user" in last_logs_str:
            st.session_state.training_active = False
            st.info(
                f"⏹️ Training stopped by user. Elapsed time: {format_elapsed_time(total_time)}")
        elif not training_flag_active:
            st.session_state.training_active = False
            st.success(
                f"✅ Training completed! Total time: {format_elapsed_time(total_time)}")
    elif not training_flag_active:
        st.session_state.training_active = False
        st.success(
            f"✅ Training completed! Total time: {format_elapsed_time(total_time)}")


def _render_active_training_ui():
    """Render UI for active training with enhanced visuals."""
    if "progress_data" in st.session_state:
        progress_data = st.session_state.progress_data
        with st.session_state.training_lock:
            current_iter = progress_data.get("iter", 0)
            current_loss = progress_data.get("loss", 0.0)
            running_loss = progress_data.get("running_loss", 0.0)
            val_loss = progress_data.get("val_loss")
            progress = progress_data.get("progress", 0.0)

        # Enhanced header with status indicator
        status_col1, status_col2 = st.columns([3, 1])
        with status_col1:
            st.header("📊 Training Progress")
        with status_col2:
            st.markdown("""
            <div style='background-color: #28a745; color: white; padding: 8px 16px; 
                        border-radius: 20px; text-align: center; font-weight: bold; margin-top: 20px;'>
                🟢 Training...
            </div>
            """, unsafe_allow_html=True)

        # Progress bar with better styling
        max_iters = st.session_state.trainer.max_iters if st.session_state.trainer else '?'
        st.progress(
            progress, text=f"Iteration {current_iter:,} / {max_iters:,}")

        # Enhanced metrics - Timing first, then Performance below
        render_training_metrics(
            current_iter=current_iter,
            current_loss=current_loss,
            running_loss=running_loss,
            val_loss=val_loss,
            progress=progress,
            max_iters=max_iters
        )

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
        render_all_losses_graph(all_losses_data, training_type="Training")

    if loss_data["iterations"]:
        render_eval_losses_graph(loss_data)
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
        has_error = any(
            "Error" in log or "ERROR" in log for log in training_logs)
        with st.expander("View All Logs", expanded=has_error):
            log_text = "\n".join(training_logs)
            st.text_area("Logs", value=log_text, height=400,
                         label_visibility="collapsed", disabled=True)
        st.caption(f"Showing {len(training_logs)} log entries")


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
        render_completed_training_ui(training_type="Training")


def _render_quick_stats(model_config, batch_size, lr, epochs):
    """Render quick statistics about the training configuration."""
    # Calculate estimated parameters
    d_model = model_config["d_model"]
    n_layers = model_config["n_layers"]
    d_mlp = model_config["d_mlp"]

    # Rough parameter estimate
    attn_params = n_layers * 4 * (d_model * d_model)  # Q, K, V, O
    mlp_params = n_layers * 2 * (d_model * d_mlp)  # in, out
    embed_params = d_model * 10000  # rough vocab estimate
    total_params = (attn_params + mlp_params + embed_params) / 1e6

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Est. Parameters", f"{total_params:.1f}M")
    with col2:
        st.metric("Batch Size", batch_size)
    with col3:
        st.metric("Learning Rate", f"{lr:.5f}")
    with col4:
        st.metric("Epochs", epochs)


st.title("🚂 Pre-Training")

# Initialize training state
initialize_training_state()

# File upload
with st.container():
    st.markdown("### 📁 1. Upload Training Data")
    uploaded_file = st.file_uploader(
        "Upload a text file for training",
        type=["txt"],
        help="Upload a text file to train the model on. If no file is uploaded, the default training.txt file will be used."
    )
    st.divider()

# Model configuration UI
with st.container():
    st.markdown("### ⚙️ 2. Model Architecture")
    model_config = render_model_config_ui()
    st.divider()

use_einops = st.checkbox("Use einops (recommended)", value=True)
model_config["use_einops"] = use_einops  # Store in config for code snippets

# Tokenizer selection
with st.container():
    st.markdown("### 🔤 3. Tokenizer")
    tokenizer_options = ["character", "bpe-simple",
                         "bpe-tiktoken", "sentencepiece"]
    current_tokenizer = model_config.get("tokenizer_type", "bpe-tiktoken")
    tokenizer_index = tokenizer_options.index(
        current_tokenizer) if current_tokenizer in tokenizer_options else 2
    tokenizer_type = st.selectbox(
        "Tokenizer Type",
        tokenizer_options,
        index=tokenizer_index,
        help="Character: simple but large vocab. BPE-simple: basic BPE implementation (educational). BPE-tiktoken: subword units using tiktoken (GPT-2 style). SentencePiece: multilingual support (LLaMA/OLMo style)."
    )
    model_config["tokenizer_type"] = tokenizer_type
    st.divider()

# Hyperparameters
with st.container():
    st.markdown("### 🎛️ 4. Training Hyperparameters")

    tab1, tab2, tab3 = st.tabs(
        ["📊 Core Settings", "🎯 Optimization", "💾 Checkpointing"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input(
                "Batch Size", min_value=1, max_value=128, value=32,
                help="Number of samples per batch")
        with col2:
            epochs = st.number_input(
                "Epochs", min_value=1, max_value=100, value=10,
                help="Number of training epochs")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            learning_rate = st.number_input(
                "Learning Rate", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f",
                help="Initial learning rate")
        with col2:
            weight_decay = st.number_input(
                "Weight Decay", min_value=0.0, max_value=1.0, value=1e-2, format="%.5f",
                help="L2 regularization strength")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            eval_interval = st.number_input(
                "Evaluation Interval", min_value=100, max_value=5000, value=500,
                help="Evaluate every N iterations")
        with col2:
            save_interval = st.number_input(
                "Save Interval", min_value=100, max_value=5000, value=1000,
                help="Save checkpoint every N iterations")

    max_steps_per_epoch = st.number_input(
        "Max Steps per Epoch", min_value=100, max_value=10000, value=500,
        help="Maximum number of training steps per epoch")

    # Quick stats
    _render_quick_stats(model_config, batch_size, learning_rate, epochs)
    st.divider()

# Understand Your Model
with st.container():
    st.markdown("### 📚 5. Understand Your Model")

    # Show architecture diagram
    render_model_architecture_diagram(model_config)

    # Show mathematical equations
    render_model_equations(model_config)

    # Show code implementation
    render_model_code_snippets(model_config)
    st.divider()

# Start training button
with st.container():
    st.markdown("### 🚀 6. Start Training")

    col1, col2, col3, col4 = st.columns([2, 1, 1, 2])

    with col2:
        start_training = st.button(
            "🚀 Start Training", type="primary", use_container_width=True,
            help="Begin training with current configuration")
    with col3:
        stop_training = st.button(
            "⏹️ Stop Training", use_container_width=True,
            help="Stop the current training run",
            disabled=not st.session_state.training_active)

    # Configuration summary before starting
    if start_training:
        with st.expander("📋 Configuration Summary", expanded=True):
            st.json({
                "Model": model_config,
                "Hyperparameters": {
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "epochs": epochs,
                    "max_steps_per_epoch": max_steps_per_epoch,
                    "eval_interval": eval_interval,
                    "save_interval": save_interval
                },
                "Tokenizer": tokenizer_type,
                "Use Einops": use_einops
            })
    st.divider()

if stop_training and st.session_state.training_active:
    with st.session_state.training_lock:
        if "training_active_flag" in st.session_state:
            st.session_state.training_active_flag[0] = False
    if "training_start_time" in st.session_state and "training_end_time" not in st.session_state:
        st.session_state.training_end_time = time.time()
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
