"""Fine-tuning page for supervised fine-tuning (SFT)."""

import os
import time
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from pretraining.tokenization.tokenizer import (
    CharacterTokenizer,
    SimpleBPETokenizer,
    BPETokenizer,
    SentencePieceTokenizer,
)
from finetuning.data.sft_dataset import SFTDataset
from finetuning.training.sft_trainer import SFTTrainer
from finetuning.training.finetuning_args import FinetuningArgs
from pretraining.training.training_ui import initialize_training_state
from ui_components import (
    render_checkpoint_selector, render_finetuning_equations, render_finetuning_code_snippets,
    format_elapsed_time, render_training_metrics,
    render_all_losses_graph, render_eval_losses_graph,
    display_training_status, render_attention_heatmap
)
from config import PositionalEncoding
from utils import get_device


def _extend_positional_embeddings(pos_embed_module, new_max_length: int):
    """
    Extend positional embeddings to support longer sequences.
    
    Uses interpolation to extend the embedding matrix. This is a common
    technique for extending pre-trained positional embeddings to longer contexts.
    
    Args:
        pos_embed_module: The positional embedding module (PosEmbedWithEinops or PosEmbedWithoutEinops)
        new_max_length: New maximum sequence length
    """
    old_W_pos = pos_embed_module.W_pos  # [old_n_ctx, d_model]
    old_n_ctx, d_model = old_W_pos.shape

    if new_max_length <= old_n_ctx:
        # No extension needed
        return

    # Create new embedding matrix
    new_W_pos = torch.empty((new_max_length, d_model),
                            device=old_W_pos.device, dtype=old_W_pos.dtype)

    # Copy existing embeddings
    new_W_pos[:old_n_ctx] = old_W_pos

    # For positions beyond the original length, use interpolation
    # Method: Use the last few positions to extrapolate smoothly
    if old_n_ctx >= 2:
        # Use the trend from the last few positions
        # Compute average "velocity" (difference between consecutive positions)
        # and extrapolate
        last_few = min(10, old_n_ctx)  # Use last 10 positions or all if fewer
        recent_embeds = old_W_pos[-last_few:]  # [last_few, d_model]

        # Compute average change per position
        if last_few >= 2:
            diffs = recent_embeds[1:] - \
                recent_embeds[:-1]  # [last_few-1, d_model]
            # [d_model] - average change per position
            avg_diff = diffs.mean(dim=0)
        else:
            avg_diff = torch.zeros_like(old_W_pos[-1])

        # Extrapolate: start from last position and add scaled differences
        last_embed = old_W_pos[-1]  # [d_model]
        for i in range(old_n_ctx, new_max_length):
            # Scale the difference based on how far we are from the original range
            # Use a decay factor to prevent embeddings from growing too large
            steps_from_end = i - old_n_ctx + 1
            decay = 0.9 ** steps_from_end  # Exponential decay
            new_W_pos[i] = last_embed + avg_diff * steps_from_end * decay
    else:
        # If only one position, just repeat it
        new_W_pos[old_n_ctx:] = old_W_pos[-1]

    # Update the parameter
    pos_embed_module.W_pos = torch.nn.Parameter(new_W_pos)
    # Update cfg.n_ctx to reflect the new max length
    pos_embed_module.cfg.n_ctx = new_max_length


def _render_quick_stats(batch_size, lr, epochs, max_length, use_lora, lora_rank=None):
    """Render quick statistics about the fine-tuning configuration."""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Batch Size", batch_size)
    with col2:
        st.metric("Learning Rate", f"{lr:.6f}")
    with col3:
        st.metric("Epochs", epochs)
    with col4:
        st.metric("Max Length", max_length)

    if use_lora and lora_rank:
        st.info(
            f"💡 Using LoRA with rank {lora_rank} (parameter-efficient fine-tuning)")


st.title("🎯 Fine-Tuning (SFT)")

# Initialize training state
initialize_training_state()

# Checkpoint selection
with st.container():
    selected_checkpoint = render_checkpoint_selector(
        header="1. Select Pre-Trained Checkpoint",
        filter_finetuned=True,
        help_text="Select a pre-trained model to fine-tune",
        no_checkpoints_message="No pre-trained checkpoints found. Please pre-train a model first."
    )
    st.divider()

# Fine-tuning method selection
with st.container():
    st.markdown("### 🔧 2. Fine-Tuning Method")
    fine_tuning_method = st.radio(
        "Select fine-tuning method",
        ["Full Parameter Fine-Tuning", "LoRA (Parameter-Efficient)"],
        help="Full Parameter: Updates all model weights. LoRA: Only trains small adapter matrices (faster, less memory)."
    )
    st.divider()

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
with st.container():
    st.markdown("### 📁 3. Upload Training Data")
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
            st.dataframe(df.head(), width='stretch')
            st.caption(f"Total rows: {len(df)}")
        else:
            st.warning(
                "No CSV file uploaded and finetuning.csv not found. Please upload a CSV file.")
    else:
        # Preview uploaded CSV
        df = pd.read_csv(uploaded_csv)
        st.dataframe(df.head(), width='stretch')
        st.caption(f"Total rows: {len(df)}")
    st.divider()

# Hyperparameters
with st.container():
    st.markdown("### 🎛️ 4. Fine-Tuning Hyperparameters")

    tab1, tab2, tab3 = st.tabs(
        ["📊 Core Settings", "🎯 Optimization", "💾 Checkpointing"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input(
                "Batch Size", min_value=1, max_value=32, value=4,
                help="Number of samples per batch")
        with col2:
            epochs = st.number_input("Epochs", min_value=1, max_value=10, value=3,
                                     help="Number of training epochs")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            learning_rate = st.number_input(
                "Learning Rate", min_value=1e-6, max_value=1e-3, value=1e-5, format="%.6f",
                help="Initial learning rate (typically 10-100x lower than pre-training)")
        with col2:
            weight_decay = st.number_input(
                "Weight Decay", min_value=0.0, max_value=1.0, value=0.01, format="%.5f",
                help="L2 regularization strength")

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            eval_interval = st.number_input(
                "Evaluation Interval", min_value=10, max_value=2000, value=50,
                help="Evaluate every N iterations")
        with col2:
            save_interval = st.number_input(
                "Save Interval", min_value=100, max_value=2000, value=500,
                help="Save checkpoint every N iterations")

    max_steps_per_epoch = st.number_input(
        "Max Steps per Epoch", min_value=10, max_value=5000, value=200,
        help="Maximum number of training steps per epoch")

    max_length = st.number_input(
        "Max Sequence Length", min_value=128, max_value=2048, value=512,
        help="Maximum length for prompt+response sequences")

    # Quick stats
    _render_quick_stats(batch_size, learning_rate, epochs,
                        max_length, use_lora, lora_rank if use_lora else None)
    st.divider()

# Understand Your Model
with st.container():
    st.markdown("### 📚 5. Understand Your Model")

    # Show equations (after LoRA config so values are available)
    render_finetuning_equations(
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

    # Show code implementation
    render_finetuning_code_snippets(use_lora=use_lora)
    st.divider()

# Start fine-tuning button
with st.container():
    st.markdown("### 🚀 6. Start Fine-Tuning")

    col1, col2, col3, col4 = st.columns([1.5, 1, 1, 1.5])
    
    # Initialize session if needed
    if "sft_trainer" not in st.session_state:
        st.session_state.sft_initialized = False
        
    # Unified Control Logic
    is_initialized = st.session_state.get("sft_initialized", False)
    is_auto_stepping = st.session_state.get("sft_auto_stepping", False)
    
    # Dynamic Label
    if not is_initialized:
        btn_label = "▶️ Start Fine-Tuning"
        btn_type = "primary"
    elif is_auto_stepping:
        btn_label = "⏸️ Pause"
        btn_type = "secondary"
    else:
        btn_label = "▶️ Resume"
        btn_type = "primary"
        
    def toggle_run_state():
        if st.session_state.get("sft_initialized", False):
            st.session_state.sft_auto_stepping = not st.session_state.sft_auto_stepping

    with col2:
        # Unified Button
        unified_btn = st.button(
            btn_label, 
            type=btn_type, 
            width='stretch',
            on_click=toggle_run_state if is_initialized else None,
            key="btn_sft_unified_control"
        )
        
    # Initialization Logic
    if unified_btn and not is_initialized:
         if not uploaded_csv and not os.path.exists("finetuning.csv"):
             st.error("Please upload a CSV file or ensure finetuning.csv exists.")
         elif not selected_checkpoint:
             st.error("Please select a pre-trained checkpoint.")
         else:
             with st.spinner("Initializing Fine-Tuning State..."):
                 device = get_device()
                 
                 # 1. Load Model
                 model, cfg, checkpoint = st.session_state.load_model_from_checkpoint(
                     selected_checkpoint["path"], device
                 )
                 model.train()
                 
                 # 2. Check Sequence Length
                 model_max_length = cfg.n_ctx if hasattr(cfg, 'n_ctx') else 256
                 if max_length > model_max_length:
                     uses_learned_pos = (
                         hasattr(cfg, 'positional_encoding') and
                         cfg.positional_encoding == PositionalEncoding.LEARNED
                     )
                     if uses_learned_pos:
                         if hasattr(model, 'pos_embed') and model.pos_embed is not None:
                             _extend_positional_embeddings(model.pos_embed, max_length)
                             st.toast(f"Extended pos embeddings to {max_length}", icon="📏")
                         else:
                             st.error(f"Cannot extend pos embeddings. Max length {model_max_length}")
                             st.stop()
                 
                 # 3. Apply LoRA
                 if use_lora:
                     from finetuning.peft.lora_utils import convert_model_to_lora
                     model = convert_model_to_lora(
                         model, rank=lora_rank, alpha=lora_alpha, 
                         dropout=lora_dropout, target_modules=lora_target_modules
                     )
                     model = model.to(device)
                 
                 # 4. Tokenizer
                 tokenizer_type = checkpoint.get("tokenizer_type", "character")
                 if tokenizer_type == "character":
                    if os.path.exists("training.txt"):
                        with open("training.txt", "r", encoding="utf-8") as f:
                            text = f.read()
                        tokenizer = CharacterTokenizer(text)
                    else:
                        st.error("training.txt needed for char tokenizer.")
                        st.stop()
                 elif tokenizer_type == "bpe-simple":
                    if os.path.exists("training.txt"):
                        with open("training.txt", "r", encoding="utf-8") as f:
                            text = f.read()
                        vocab_size = cfg.d_vocab if hasattr(cfg, 'd_vocab') else 1000
                        tokenizer = SimpleBPETokenizer(text, vocab_size=vocab_size)
                    else:
                        st.error("training.txt needed for simple BPE.")
                        st.stop()
                 elif tokenizer_type in ["bpe-tiktoken", "bpe"]:
                     tokenizer = BPETokenizer()
                 elif tokenizer_type == "sentencepiece":
                    if os.path.exists("training.txt"):
                        with open("training.txt", "r", encoding="utf-8") as f:
                            text = f.read()
                        vocab_size = cfg.d_vocab if hasattr(cfg, 'd_vocab') else 10000
                        tokenizer = SentencePieceTokenizer(text, vocab_size=vocab_size)
                 else:
                     st.error(f"Unknown tokenizer: {tokenizer_type}")
                     st.stop()

                 # 5. Data
                 csv_path = "temp_sft_data.csv"
                 if uploaded_csv:
                     with open(csv_path, "wb") as f:
                         f.write(uploaded_csv.getbuffer())
                 else:
                     import shutil
                     shutil.copy("finetuning.csv", csv_path)
                 
                 dataset = SFTDataset(csv_path=csv_path, tokenizer=tokenizer, max_length=max_length)
                 X_train, Y_train, masks_train = dataset.get_train_data()
                 X_val, Y_val, masks_val = dataset.get_val_data()
                 
                 # 6. Args
                 checkpoint_dir = Path(selected_checkpoint["path"]).parent
                 sft_dir = checkpoint_dir / "sft"
                 sft_dir.mkdir(exist_ok=True)
                 
                 training_args = FinetuningArgs(
                     batch_size=batch_size, epochs=epochs, max_steps_per_epoch=max_steps_per_epoch,
                     lr=learning_rate, weight_decay=weight_decay, save_dir=str(sft_dir),
                     save_interval=save_interval, eval_iters=50, use_lora=use_lora,
                     lora_rank=lora_rank if use_lora else 8,
                     lora_alpha=lora_alpha if use_lora else 8.0,
                     lora_dropout=lora_dropout if use_lora else 0.0,
                     lora_target_modules=lora_target_modules if use_lora else "all"
                 )
                 
                 # 7. Trainer
                 st.session_state.sft_trainer = SFTTrainer(
                     model, training_args, X_train, Y_train, masks_train,
                     X_val, Y_val, masks_val, device, eval_interval=eval_interval,
                     tokenizer_type=tokenizer_type
                 )
                 st.session_state.sft_tokenizer = tokenizer
                 st.session_state.sft_initialized = True
                 st.session_state.sft_logs = []
                 st.session_state.sft_auto_stepping = True
                 st.session_state.sft_start_time = time.time()
                 st.session_state.sft_shared_loss_data = {
                     "iterations": [], "train_losses": [], "val_losses": []
                 }
                 
                 st.success("Fine-Tuning Started!")
                 st.rerun()

    # Step Progress Logic
    if st.session_state.get("sft_initialized", False) and "sft_logs" in st.session_state:
        current_step = len(st.session_state.sft_logs)
        total_steps = st.session_state.sft_trainer.max_iters
        progress = min(current_step / total_steps, 1.0)
        st.progress(progress, text=f"Progress: Batch {current_step} / {total_steps}")
    
    # Ensure auto_stepping is initialized
    if "sft_auto_stepping" not in st.session_state:
        st.session_state.sft_auto_stepping = False
    
    is_auto_stepping = st.session_state.sft_auto_stepping
    
    with col3:
         step_btn = st.button("⏭️ Step", width='stretch',
                             disabled=not st.session_state.get("sft_initialized", False),
                             key="btn_sft_step")
                             
    # Logic for performing a step
    should_step = False
    if step_btn and st.session_state.get("sft_initialized", False):
        should_step = True
    elif is_auto_stepping and st.session_state.get("sft_initialized", False):
        should_step = True
        
    if should_step:
        # Check if done
        current_len = len(st.session_state.sft_logs)
        max_len = st.session_state.sft_trainer.max_iters
        
        if current_len >= max_len:
            st.session_state.sft_auto_stepping = False
            trainer = st.session_state.sft_trainer
            trainer.save_checkpoint(max_len, is_final=True)
            st.success("Fine-tuning Complete! Model saved.")
        else:
            trainer = st.session_state.sft_trainer
            metrics = trainer.train_single_step()
            st.session_state.sft_logs.append(metrics)
            st.session_state.last_sft_metrics = metrics
            
            # Print to CLI
            print(f"SFT Iter {len(st.session_state.sft_logs)}: loss {metrics['loss']:.4f}", flush=True)
            
            # Evaluate
            curr_iter = len(st.session_state.sft_logs)
            if curr_iter % trainer.eval_interval == 0:
                losses = trainer.estimate_loss()
                val_loss = losses["val"]
                print(f"EVAL: step {curr_iter}, val_loss {val_loss:.4f}", flush=True)
                st.session_state.sft_shared_loss_data["iterations"].append(curr_iter)
                st.session_state.sft_shared_loss_data["train_losses"].append(metrics["loss"])
                st.session_state.sft_shared_loss_data["val_losses"].append(val_loss)
            
            # Save Checkpoint
            if hasattr(trainer.args, "save_interval") and curr_iter % trainer.args.save_interval == 0:
                trainer.save_checkpoint(curr_iter)
    
    # Display Metrics & Visuals
    if "last_sft_metrics" in st.session_state:
        metrics = st.session_state.last_sft_metrics
        current_step = len(st.session_state.sft_logs)
        max_steps = st.session_state.sft_trainer.max_iters
        progress = min(current_step / max_steps, 1.0)
        
        # 1. Metrics
        latest_val_loss = None
        if st.session_state.sft_shared_loss_data["val_losses"]:
            latest_val_loss = st.session_state.sft_shared_loss_data["val_losses"][-1]
            
        render_training_metrics(
            current_iter=current_step,
            current_loss=metrics["loss"],
            running_loss=metrics["running_loss"],
            val_loss=latest_val_loss,
            progress=progress,
            max_iters=max_steps
        )
        
        # 2. Graph
        all_losses_data = {
            "iterations": list(range(1, current_step + 1)),
            "current_losses": [m["loss"] for m in st.session_state.sft_logs],
            "running_losses": [m["running_loss"] for m in st.session_state.sft_logs]
        }
        render_all_losses_graph(all_losses_data, training_type="Fine-Tuning Interactive")
        
        if st.session_state.sft_shared_loss_data["iterations"]:
            render_eval_losses_graph(st.session_state.sft_shared_loss_data)
            
        # 3. Text Samples
        if "inputs" in metrics and "targets" in metrics:
            st.markdown("### 🔍 Inspect Batch")
            current_bs = metrics["inputs"].shape[0]
            sample_idx = st.slider("Select Sample", 1, current_bs, 1) - 1
            
            input_ids = metrics["inputs"][sample_idx]
            target_ids = metrics["targets"][sample_idx]
            masks = metrics["masks"][sample_idx]
            
            # Decode with color highlighting for Prompt vs Response (based on mask)
            # Mask = 0 -> Prompt (ignore loss)
            # Mask = 1 -> Response (compute loss)
            # Mask = 0 (after response) -> Padding (ignore loss)
            
            tokens_html = ""
            import html
            tokenizer = st.session_state.sft_tokenizer
            
            # Identify effective sequence length (Prompt + Response, ignoring Padding)
            try:
                # Find last index where mask is 1 (end of response)
                last_response_idx = (masks == 1).nonzero(as_tuple=True)[0][-1].item()
                # We want to show everything up to this index
                effective_len = last_response_idx + 1
            except IndexError:
                # Fallback
                effective_len = len(masks)
            
            # Slice the data for visualization
            input_ids_view = input_ids[:effective_len]
            masks_view = masks[:effective_len]
            
            token_labels = [] # For heatmap
            
            for i, tid in enumerate(input_ids_view.tolist()):
                is_response = masks_view[i].item() == 1
                
                if is_response:
                    bg_color = "rgba(78, 205, 196, 0.4)" # Teal
                    border = "1px solid #4CAF50"
                    status = "Response"
                else: # Prompt
                    bg_color = "rgba(255, 255, 255, 0.1)" # Grey
                    border = "1px solid #555"
                    status = "Prompt"
                
                token_text = tokenizer.decode([tid])
                token_labels.append(f"'{token_text}'")
                safe_token = html.escape(token_text)
                
                tokens_html += f'<span style="background-color: {bg_color}; border: {border}; margin: 0 1px; padding: 0 2px;" title="{status} | ID: {tid}">{safe_token}</span>'
                
            st.markdown(
                f'<div style="background-color: #262730; padding: 15px; border-radius: 5px; line-height: 1.8; font-family: monospace;">{tokens_html}</div>', 
                unsafe_allow_html=True
            )
            st.caption("Teal = Response (Target), Grey = Prompt. Padding tokens are hidden.")
            
            # Attention Heatmap
            st.divider()
            with st.expander("🔥 Attention Heatmaps", expanded=True):
                 col_l, col_h = st.columns(2)
                 cfg = st.session_state.sft_trainer.model.cfg
                 with col_l:
                     layer_idx = st.slider("Layer", 0, cfg.n_layers - 1, 0, key=f"sft_attn_l_{current_step}")
                 with col_h:
                     head_idx = st.slider("Head", 0, cfg.n_heads - 1, 0, key=f"sft_attn_h_{current_step}")
                 
                 # Compute diagnostics
                 with st.spinner("Calculating attention..."):
                     # Pass full padded sequence to model for correct calculation
                     sample_tokens = input_ids.unsqueeze(0).to(get_device())
                     model = st.session_state.sft_trainer.model
                     with torch.no_grad():
                         outputs = model(sample_tokens, return_diagnostics=True)
                         diagnostics = outputs[-1] if isinstance(outputs, tuple) else None
                 
                 if diagnostics and "attention_patterns" in diagnostics:
                     # Slice the attention map to ignore padding
                     # Pattern shape: [1, n_heads, seq_len, seq_len] -> [seq_len, seq_len] for specific head
                     full_attn_map = diagnostics["attention_patterns"][layer_idx][0, head_idx].cpu().numpy()
                     
                     # Visualize only the non-padded region
                     attn_map = full_attn_map[:effective_len, :effective_len]
                     
                     render_attention_heatmap(attn_map, token_labels, layer_idx, head_idx)
    
    # Auto-stepping trigger
    if st.session_state.get("sft_auto_stepping", False) and st.session_state.get("sft_initialized", False):
        st.rerun()

# Display training status
display_training_status(training_type="Fine-Tuning")
