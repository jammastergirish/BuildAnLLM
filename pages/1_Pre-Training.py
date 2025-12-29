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
from pretraining.training.training_ui import initialize_training_state, train_model_thread
from utils import get_device, format_elapsed_time

from ui_components import (
    render_model_config_ui, render_model_architecture_diagram, render_model_equations,
    render_model_code_snippets, format_elapsed_time, get_total_training_time,
    render_training_metrics, render_all_losses_graph, render_eval_losses_graph,
    render_completed_training_ui, render_active_training_ui, display_training_status,
    render_attention_heatmap
)


# Define helper functions first
def _create_model_config(model_config: dict) -> ModelConfig:
    """Create ModelConfig from UI config dict."""
    from config import RouterType

    cfg = ModelConfig(
        architecture=Architecture.GPT,  # Base, doesn't matter
        d_model=model_config["d_model"],
        n_heads=model_config["n_heads"],
        # Default to MHA if not specified
        n_kv_heads=model_config.get("n_kv_heads", model_config["n_heads"]),
        n_layers=model_config["n_layers"],
        n_ctx=model_config["n_ctx"],
        d_head=model_config["d_head"],
        d_mlp=model_config["d_mlp"],
        positional_encoding=PositionalEncoding(
            model_config["positional_encoding"]),
        normalization=Normalization(model_config["normalization"]),
        activation=Activation(model_config["activation"]),
        rope_theta=model_config.get("rope_theta", 10000.0),
        use_moe=model_config.get("use_moe", False),
        num_experts=model_config.get("num_experts", 8),
        num_experts_per_tok=model_config.get("num_experts_per_tok", 2),
        use_shared_experts=model_config.get("use_shared_experts", False),
        num_shared_experts=model_config.get("num_shared_experts", 2),
        router_type=RouterType(model_config.get(
            "router_type", "top_k")) if model_config.get("use_moe", False) else None,
        load_balancing_loss_weight=model_config.get(
            "load_balancing_loss_weight", 0.01),
        expert_capacity_factor=model_config.get(
            "expert_capacity_factor", 1.25),
    )
    return cfg


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
    device = get_device()
    from pretraining.model.model import TransformerModel
    model = TransformerModel(cfg, use_einops=use_einops)
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
             cfg = _create_model_config(model_config)
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
                 device=device, tokenizer_type=tokenizer_type
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
         max_steps = st.session_state.manual_trainer.max_iters
         progress = min(current_step / max_steps, 1.0)
         
         # 1. Render Standard Metrics
         # Handle val_loss display if available
         latest_val_loss = None
         if st.session_state.shared_loss_data["val_losses"]:
             latest_val_loss = st.session_state.shared_loss_data["val_losses"][-1]
             
         render_training_metrics(
             current_iter=current_step,
             current_loss=metrics["loss"],
             running_loss=metrics["running_loss"],
             val_loss=latest_val_loss, 
             progress=progress,
             max_iters=max_steps
         )
         
         # 2. Render Standard Graph (Training Loss)
         all_losses_data = {
             "iterations": list(range(1, current_step + 1)),
             "current_losses": [m["loss"] for m in st.session_state.manual_logs],
             "running_losses": [m["running_loss"] for m in st.session_state.manual_logs]
         }
         render_all_losses_graph(all_losses_data, training_type="Interactive Training")
         
         # 2b. Render Evaluation Graph (Train vs Val) - IF we have data
         if st.session_state.shared_loss_data["iterations"]:
             render_eval_losses_graph(st.session_state.shared_loss_data)
         
         # 3. Render Text Samples
         if "inputs" in metrics and "targets" in metrics and "manual_tokenizer" in st.session_state:
             # Get batch size from data
             current_bs = metrics["inputs"].shape[0]
             n_ctx = metrics["inputs"].shape[1]
             
             # Allow user to pick which sample in the batch to view
             sample_idx = st.slider(
                 "Inspect Batch Sample", 
                 min_value=1, 
                 max_value=current_bs, 
                 value=1,
                 help="Select which sequence from the current batch to inspect."
             ) - 1 # Convert 1-based UI to 0-based index
             
             # Decode selected sample
             input_ids = metrics["inputs"][sample_idx]
             target_ids = metrics["targets"][sample_idx]
             
             input_ids_list = input_ids.tolist()
             
             # Generate Colored Tokens HTML
             colored_html = ""
             # Palette of translucent colors for dark mode
             color_palette = [
                 "rgba(255, 107, 107, 0.4)",   # Red
                 "rgba(78, 205, 196, 0.4)",    # Teal
                 "rgba(255, 217, 61, 0.4)",    # Yellow
                 "rgba(167, 139, 250, 0.4)",   # Purple
                 "rgba(255, 159, 26, 0.4)",    # Orange
                 "rgba(69, 170, 242, 0.4)",    # Blue
             ]
             
             import html
             
             for i, token_id in enumerate(input_ids_list):
                 # Decode individual token
                 token_text = st.session_state.manual_tokenizer.decode([token_id])
                 # Handle special characters for HTML
                 safe_token = html.escape(token_text)
                 # Choose color
                 color = color_palette[i % len(color_palette)]
                 
                 # Wrap in span. Use a title attribute to show the Token ID on hover!
                 colored_html += f'<span style="background-color: {color}; border-radius: 2px; padding: 0 1px;" title="ID: {token_id}">{safe_token}</span>'
             
             # Format Target: Just show the LAST token (the "next" token for the full sequence)
             last_token_id = target_ids[-1].item()
             last_token_text = st.session_state.manual_tokenizer.decode([last_token_id])
             target_display = f"{last_token_text}"
             
             st.markdown(f"##### 📖 Current Batch Sample ({sample_idx + 1} of {current_bs})")
             st.caption(f"Sequence Length: {n_ctx} tokens (defined by n_ctx)")
             
             c1, c2 = st.columns([3, 1])
             with c1:
                 st.markdown("**Input (Context)**")
                 st.markdown(
                     f'<div style="background-color: #262730; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; font-size: 14px; line-height: 1.8;">{colored_html}</div>',
                     unsafe_allow_html=True
                 )
             with c2:
                 st.markdown("**Target (Next Token)**")
                 st.markdown(
                     f'<div style="background-color: #262730; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; font-size: 14px; border: 1px solid #4CAF50;">{target_display}</div>',
                     unsafe_allow_html=True
                 )
                 st.caption("The model predicts the next token at every position. Here we show the final target token.")

             # === ATTENTION HEATMAPS ===
             with st.spinner("Calculating attention patterns..."):
                 # We need to run a forward pass on this specific sample to get diagnostics
                 # input_ids is [seq_len], need to unsqueeze to [1, seq_len]
                 sample_tokens = input_ids.unsqueeze(0).to(get_device())
                 
                 # Use the manual trainer's model
                 model = st.session_state.manual_trainer.model
                 
                 with torch.no_grad():
                     outputs = model(sample_tokens, return_diagnostics=True)
                     if isinstance(outputs, tuple):
                         diagnostics = outputs[-1]
                     else:
                         diagnostics = None
                         
                 if diagnostics and "attention_patterns" in diagnostics:
                     st.divider()
                     st.markdown("### 🔥 Attention Heatmaps")
                     st.caption("Visualize how the model attends to different tokens in this sample.")
                     
                     # Select Layer and Head
                     col_l, col_h = st.columns(2)
                     cfg = model.cfg
                     with col_l:
                         layer_idx = st.slider("Select Layer", 0, cfg.n_layers - 1, 0, key=f"attn_layer_{current_step}")
                     with col_h:
                         head_idx = st.slider("Select Head", 0, cfg.n_heads - 1, 0, key=f"attn_head_{current_step}")
                     
                     # Get attention pattern
                     attn_map = diagnostics["attention_patterns"][layer_idx][0, head_idx].cpu().numpy()
                     
                     # Labels (we already have token texts from above loop, but let's reconstruct clean list)
                     token_labels = []
                     for tid in input_ids_list:
                         try:
                             decoded = st.session_state.manual_tokenizer.decode([tid])
                             # Escape for display if needed, but for plot labels usually raw is fine unless it breaks stuff
                             token_labels.append(f"'{decoded}'")
                         except:
                             token_labels.append(f"T{tid}")
                             
                     
                     # Render heatmap using shared component
                     render_attention_heatmap(attn_map, token_labels, layer_idx, head_idx)
                 
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
