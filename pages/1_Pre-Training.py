"""Pre-training page for transformer models."""

import streamlit as st
import os
import threading
import time
import pandas as pd
import plotly.express as px
import torch
from datetime import datetime

from config import ModelConfig, Architecture, PositionalEncoding, Normalization, Activation
from pretraining.training.training_args import TransformerTrainingArgs
from pretraining.training.trainer import TransformerTrainer
from pretraining.data.dataset import TransformerDataset
from pretraining.model.model import TransformerModel
from training_utils import initialize_training_state
from utils import get_device, format_elapsed_time

from ui_components import (
    render_model_config_ui, render_model_architecture_diagram, render_model_equations,
    render_model_code_snippets, format_elapsed_time, get_total_training_time,
    render_training_metrics, render_all_losses_graph, render_eval_losses_graph,
    render_completed_training_ui, render_active_training_ui, display_training_status,
    render_attention_heatmap, render_interactive_dashboard
)


# Define helper functions first






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

    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1.5])
    
    # Initialize session if needed
    if "manual_trainer" not in st.session_state:
        st.session_state.manual_initialized = False
        
    # Unified Control Logic
    is_initialized = st.session_state.get("manual_initialized", False)
    is_auto_stepping = st.session_state.get("auto_stepping", False)
    
    # Dynamic Label
    if not is_initialized:
        btn_label = "▶️ Start Training"
        btn_type = "primary"
    elif is_auto_stepping:
        btn_label = "⏸️ Pause"
        btn_type = "secondary"
    else:
        btn_label = "▶️ Resume"
        btn_type = "primary"
        
    def toggle_run_state():
        if st.session_state.get("manual_initialized", False):
            st.session_state.auto_stepping = not st.session_state.auto_stepping

    with col2:
        # Unified Button
        # Logic: If NOT initialized, click -> returns True -> Runs Init Block
        # If Initialized, click -> Run Callback (toggle) -> Returns True (ignore in init block)
        unified_btn = st.button(
            btn_label, 
            type=btn_type, 
            width='stretch',
            on_click=toggle_run_state if is_initialized else None,
            key="btn_unified_control"
        )
        
    # Initialization Logic
    # Verify we need to init (Button clicked AND not initialized)
    if unified_btn and not is_initialized:
         with st.spinner("Initializing Training State..."):
             # Load text
             if uploaded_file:
                 text = uploaded_file.read().decode("utf-8")
             else:
                 with open("training.txt", "r", encoding="utf-8") as f:
                     text = f.read()
             
             # Create config/dataset/model
             cfg = ModelConfig.from_ui_dict(model_config)
             dataset = TransformerDataset(text, cfg, tokenizer_type=tokenizer_type)
             cfg = dataset.cfg
             device = get_device()
             model = TransformerModel(cfg, use_einops=use_einops).to(device)
             
             # Create timestamped checkpoint directory
             timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
             save_dir = os.path.join("checkpoints", timestamp)
             os.makedirs(save_dir, exist_ok=True)
             
             training_args = TransformerTrainingArgs(
                 batch_size=batch_size, epochs=epochs, max_steps_per_epoch=max_steps_per_epoch,
                 lr=learning_rate, weight_decay=weight_decay, eval_iters=10,
                 save_dir=save_dir, save_interval=save_interval
             )
             
             st.session_state.manual_trainer = TransformerTrainer(
                 model, training_args, *dataset.get_train_data(), *dataset.get_val_data(), 
                 device=device, tokenizer_type=tokenizer_type, tokenizer=dataset.tokenizer
             )
             st.session_state.manual_tokenizer = dataset.tokenizer # Store for decoding
             st.session_state.manual_initialized = True
             st.session_state.manual_logs = []
             st.session_state.auto_stepping = True # Default to Auto-Step
             
             # Initialize timing and eval state for the Stepper
             st.session_state.training_start_time = time.time()
             st.session_state.shared_loss_data = {
                 "iterations": [], "train_losses": [], "val_losses": []
             }
             
             st.success("Training Started! (Auto-Stepping)")
             st.rerun()

    # Step Progress Logic
    if st.session_state.get("manual_initialized", False) and "manual_logs" in st.session_state:
        current_step = len(st.session_state.manual_logs)
        total_steps = st.session_state.manual_trainer.max_iters
        progress = min(current_step / total_steps, 1.0)
        st.progress(progress, text=f"Training Progress: Batch {current_step} / {total_steps}")

    # Ensure auto_stepping is initialized
    if "auto_stepping" not in st.session_state:
        st.session_state.auto_stepping = False

    is_auto_stepping = st.session_state.auto_stepping

    with col3:
         # Step Button (Renamed)
         step_btn = st.button("⏭️ Step to Next Batch", width='stretch', 
                             disabled=not st.session_state.get("manual_initialized", False),
                             key="btn_manual_step")
    
    # Logic for performing a step
    should_step = False
    if step_btn and st.session_state.get("manual_initialized", False):
        should_step = True
    elif is_auto_stepping and st.session_state.get("manual_initialized", False):
        should_step = True
        
    if should_step:
         # Check if we are already done
         current_len = len(st.session_state.manual_logs)
         max_len = st.session_state.manual_trainer.max_iters
         
         if current_len >= max_len:
             st.session_state.auto_stepping = False
             
             # Save Final Checkpoint if not already done (naive check)
             # We can just force save.
             trainer = st.session_state.manual_trainer
             if hasattr(trainer.args, "save_dir"):
                 trainer.save_checkpoint(max_len, is_final=True)
                 trainer.save_loss_graph()
                 print(f"CHECKPOINT: Final model saved at iter {max_len}", flush=True)
                 
             st.success("Training Complete! Model saved.")
             # Do not step
         else:
             metrics = st.session_state.manual_trainer.train_single_step()
             st.session_state.manual_logs.append(metrics)
             st.session_state.last_manual_metrics = metrics # Store for display
             
             # Print to CLI (tqdm style)
             print(f"Iter {len(st.session_state.manual_logs)}: loss {metrics['loss']:.4f}, time {format_elapsed_time(time.time() - st.session_state.training_start_time)}", flush=True)
             
             # Run Evaluation if needed
             trainer = st.session_state.manual_trainer
             curr_iter = len(st.session_state.manual_logs)
             if curr_iter % trainer.eval_interval == 0:
                 losses = trainer.estimate_loss()
                 val_loss = losses["val"]
                 print(f"EVAL: step {curr_iter}, val_loss {val_loss:.4f}", flush=True)
                 
                 # Store for graph
                 st.session_state.shared_loss_data["iterations"].append(curr_iter)
                 st.session_state.shared_loss_data["train_losses"].append(metrics["loss"]) # approximate with current batch
                 st.session_state.shared_loss_data["val_losses"].append(val_loss)
             
             # Save Checkpoint if needed
             if hasattr(trainer.args, "save_interval") and curr_iter % trainer.args.save_interval == 0:
                 trainer.save_checkpoint(curr_iter)
                 print(f"CHECKPOINT: Saved at iter {curr_iter}", flush=True)

    # Display Metrics & Text (from last step)
    if "last_manual_metrics" in st.session_state:
        metrics = st.session_state.last_manual_metrics
        
        # Prepare data for standard UI
        current_step = len(st.session_state.manual_logs)
        
        from ui_components import render_interactive_dashboard
        
        render_interactive_dashboard(
            trainer=st.session_state.manual_trainer,
            metrics=metrics,
            current_step=current_step,
            start_time=st.session_state.training_start_time,
            loss_data=st.session_state.shared_loss_data,
            tokenizer=st.session_state.manual_tokenizer,
            logs=st.session_state.manual_logs,
            title="Interactive Training"
        )
            
    # Trigger next step if auto-stepping is active
    if st.session_state.get("auto_stepping", False) and st.session_state.get("manual_initialized", False):
        # No sleep - run as fast as possible
        st.rerun()
        
    # Visualize Manual Logs (Grad Norm)
    if st.session_state.get("manual_initialized", False) and st.session_state.manual_logs:
        import plotly.graph_objects as go
        
        logs_df = pd.DataFrame(st.session_state.manual_logs)
        iterations = list(range(1, len(logs_df) + 1))
        
        fig_grad = go.Figure()
        fig_grad.add_trace(go.Scatter(
            x=iterations, y=logs_df["grad_norm"],
            mode="lines", name="Gradient Norm",
            line={"color": "#FF4B4B", "width": 2}
        ))
        
        fig_grad.update_layout(
            title="Gradient Norm (Training Stability)",
            xaxis_title="Step", yaxis_title="Grad Norm",
            hovermode="x unified", height=300,
            template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly"
        )
        st.plotly_chart(fig_grad, width='stretch')

# Clean up helper function block if no longer needed
# But _create_model_config is still used by Init Manual
# _start_training_workflow can be removed or ignored.
