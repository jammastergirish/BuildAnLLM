"""Reusable Streamlit UI components."""

import inspect
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st
import torch
from utils import format_elapsed_time, get_elapsed_time, get_total_training_time, scan_checkpoints, get_device


# ============================================================================
# Training Time Utilities
# ============================================================================


# Time functions moved to utils.py



def render_training_metrics(
    current_iter: int,
    current_loss: float,
    running_loss: float,
    val_loss: Optional[float],
    progress: float,
    max_iters: int
) -> None:
    """Render training metrics with Timing first, then Performance.

    Args:
        current_iter: Current iteration number
        current_loss: Current loss value
        running_loss: Running average loss
        val_loss: Validation loss (can be None)
        progress: Training progress (0.0 to 1.0)
        max_iters: Maximum number of iterations (can be int or string like '?')
    """
    elapsed_time = get_elapsed_time()

    # Timing metrics first
    st.markdown("#### ⏱️ Timing")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Progress", f"{progress*100:.1f}%")
    with col2:
        st.metric("Elapsed Time", format_elapsed_time(elapsed_time))
    with col3:
        if elapsed_time > 0 and current_iter > 0 and isinstance(max_iters, int):
            iter_per_sec = current_iter / elapsed_time
            eta_seconds = (max_iters - current_iter) / iter_per_sec
            eta_str = format_elapsed_time(
                eta_seconds) if eta_seconds > 0 else "Calculating..."
            st.metric("Est. Time Remaining", eta_str)
        else:
            st.metric("Est. Time Remaining", "Calculating...")

    # Performance metrics below
    st.markdown("#### 📈 Performance")
    col1, col2, col3 = st.columns(3)
    with col1:
        delta_loss = None
        if running_loss > 0 and current_loss > 0:
            delta_loss = f"{running_loss - current_loss:.4f}"
        st.metric("Current Loss", f"{current_loss:.4f}", delta=delta_loss)
    with col2:
        st.metric("Running Avg Loss", f"{running_loss:.4f}")
    with col3:
        val_delta = None
        if val_loss is not None and running_loss > 0:
            val_delta = "↓" if val_loss < running_loss else "↑"
        st.metric(
            "Val Loss",
            f"{val_loss:.4f}" if val_loss is not None else "Pending...",
            delta=val_delta
        )


def render_all_losses_graph(all_losses_data: Dict, training_type: str = "Training") -> None:
    """Render all losses graph with enhanced styling.

    Args:
        all_losses_data: Dictionary with 'iterations', 'current_losses', 'running_losses'
        training_type: Type of training ("Training" or "Fine-Tuning") for title customization
    """
    import pandas as pd
    import plotly.graph_objects as go

    st.subheader("📈 Training Loss")

    # Add summary stats above graph
    if all_losses_data["current_losses"]:
        latest_loss = all_losses_data["current_losses"][-1]
        min_loss = min(all_losses_data["current_losses"])
        if len(all_losses_data["current_losses"]) > 10:
            recent_avg = sum(all_losses_data["current_losses"][-10:]) / 10
            earlier_avg = sum(all_losses_data["current_losses"][-20:-10]) / 10 if len(
                all_losses_data["current_losses"]) > 20 else recent_avg
            loss_trend = "↓ Improving" if recent_avg < earlier_avg else "→ Stable"
        else:
            loss_trend = "→ Initializing"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latest Loss", f"{latest_loss:.4f}")
        with col2:
            st.metric("Best Loss", f"{min_loss:.4f}")
        with col3:
            st.metric("Trend", loss_trend)

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

    title = f"{training_type} Losses (updated every 10 iterations)"
    fig_all.update_layout(
        title=title,
        xaxis_title="Iteration", yaxis_title="Loss",
        hovermode="x unified", height=400,
        yaxis={"range": [0, None]},
        template="plotly_dark" if st.get_option(
            "theme.base") == "dark" else "plotly"
    )
    st.plotly_chart(fig_all, width='stretch')


def render_eval_losses_graph(loss_data: Dict) -> None:
    """Render evaluation losses graph with enhanced styling.

    Args:
        loss_data: Dictionary with 'iterations', 'train_losses', 'val_losses'
    """
    import pandas as pd
    import plotly.graph_objects as go

    st.subheader("📊 Evaluation Losses (Train/Val)")

    # Add summary stats
    if loss_data["train_losses"] and loss_data["val_losses"]:
        latest_train = loss_data["train_losses"][-1]
        latest_val = loss_data["val_losses"][-1]
        gap = latest_val - latest_train
        gap_status = "✓ Good" if gap < 0.5 else "⚠️ Large gap" if gap < 1.0 else "❌ Overfitting"

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Latest Train Loss", f"{latest_train:.4f}")
        with col2:
            st.metric("Latest Val Loss", f"{latest_val:.4f}")
        with col3:
            st.metric("Train-Val Gap", f"{gap:.4f}", delta=gap_status)

    df = pd.DataFrame({
        "Iteration": loss_data["iterations"],
        "Train Loss": loss_data["train_losses"],
        "Val Loss": loss_data["val_losses"]
    })

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Iteration"], y=df["Train Loss"],
        mode="lines+markers", name="Train Loss",
        line={"color": "blue", "width": 2}
    ))
    fig.add_trace(go.Scatter(
        x=df["Iteration"], y=df["Val Loss"],
        mode="lines+markers", name="Val Loss",
        line={"color": "red", "width": 2}
    ))
    fig.update_layout(
        title="Training and Validation Loss (evaluated every 500 iterations)",
        xaxis_title="Iteration", yaxis_title="Loss",
        hovermode="x unified", height=400,
        template="plotly_dark" if st.get_option(
            "theme.base") == "dark" else "plotly"
    )
    st.plotly_chart(fig, width='stretch')


def render_completed_training_ui(training_type: str = "Training") -> None:
    """Render UI for completed training.

    Args:
        training_type: Type of training ("Training" or "Fine-Tuning") for text customization
    """
    import pandas as pd
    import plotly.graph_objects as go

    if st.session_state.loss_data["iterations"]:
        # Calculate total elapsed time
        total_time = get_total_training_time()

        header_text = f"📊 Final {training_type} Results"
        time_text = f"Total {training_type.lower()} time"
        st.header(header_text)
        if total_time > 0:
            st.markdown(f"""
            <div style='background-color: #17a2b8; color: white; padding: 12px 20px; 
                        border-radius: 8px; margin: 10px 0; font-weight: 500;'>
                ⏱️ {time_text}: <strong>{format_elapsed_time(total_time)}</strong>
            </div>
            """, unsafe_allow_html=True)

        # Final metrics summary
        if st.session_state.loss_data["train_losses"] and st.session_state.loss_data["val_losses"]:
            final_train = st.session_state.loss_data["train_losses"][-1]
            final_val = st.session_state.loss_data["val_losses"][-1]
            best_train = min(st.session_state.loss_data["train_losses"])
            best_val = min(st.session_state.loss_data["val_losses"])

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Train Loss", f"{final_train:.4f}")
            with col2:
                st.metric("Final Val Loss", f"{final_val:.4f}")
            with col3:
                st.metric("Best Train Loss", f"{best_train:.4f}")
            with col4:
                st.metric("Best Val Loss", f"{best_val:.4f}")

        df = pd.DataFrame({
            "Iteration": st.session_state.loss_data["iterations"],
            "Train Loss": st.session_state.loss_data["train_losses"],
            "Val Loss": st.session_state.loss_data["val_losses"]
        })

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Iteration"], y=df["Train Loss"],
            mode="lines+markers", name="Train Loss",
            line={"color": "blue", "width": 2}
        ))
        fig.add_trace(go.Scatter(
            x=df["Iteration"], y=df["Val Loss"],
            mode="lines+markers", name="Val Loss",
            line={"color": "red", "width": 2}
        ))
        fig.update_layout(
            title="Training and Validation Loss",
            xaxis_title="Iteration", yaxis_title="Loss",
            hovermode="x unified", height=400,
            template="plotly_dark" if st.get_option(
                "theme.base") == "dark" else "plotly"
        )
        st.plotly_chart(fig, width='stretch')


def handle_training_completion(training_flag_active: bool, training_type: str = "Training") -> None:
    """Handle training completion logic.

    Args:
        training_flag_active: Whether training is still active
        training_type: Type of training ("Training" or "Fine-Tuning") for text customization
    """
    import time

    # Record end time
    if "training_start_time" in st.session_state and "training_end_time" not in st.session_state:
        st.session_state.training_end_time = time.time()

    total_time = get_total_training_time()

    # Determine completion messages based on training type
    if training_type == "Fine-Tuning":
        complete_msg = "Fine-tuning complete!"
        completed_msg = "Fine-tuning completed!"
        error_msg = "Error during fine-tuning"
        stopped_msg = "Fine-tuning stopped by user"
        error_detail = "Fine-tuning error occurred. Check logs for details."
    else:
        complete_msg = "Training complete!"
        completed_msg = "Training completed!"
        error_msg = "Error during training"
        stopped_msg = "Training stopped by user"
        error_detail = "Training error occurred. Check logs for details."

    # Check for errors first (check all logs, not just last 3)
    if st.session_state.shared_training_logs:
        all_logs_str = " ".join(st.session_state.shared_training_logs)
        has_error = error_msg in all_logs_str or "ERROR DETECTED" in all_logs_str
        
        if has_error:
            st.session_state.training_active = False
            # Set end time for proper timing calculation
            if "training_start_time" in st.session_state and "training_end_time" not in st.session_state:
                st.session_state.training_end_time = time.time()
            st.error(f"❌ {error_detail}")
            return  # Don't show success message if there's an error
        
        # Check for completion messages
        last_logs = list(st.session_state.shared_training_logs)[-3:]
        last_logs_str = " ".join(last_logs)
        if complete_msg in last_logs_str or "Completed all" in last_logs_str:
            st.session_state.training_active = False
            st.success(
                f"✅ {completed_msg} Total time: {format_elapsed_time(total_time)}")
        elif stopped_msg in last_logs_str:
            st.session_state.training_active = False
            st.info(
                f"⏹️ {stopped_msg}. Elapsed time: {format_elapsed_time(total_time)}")
        elif not training_flag_active:
            # Only show success if thread finished and no errors detected
            st.session_state.training_active = False
            st.success(
                f"✅ {completed_msg} Total time: {format_elapsed_time(total_time)}")
    elif not training_flag_active:
        # Thread finished but no logs - check if there were any errors
        # If thread died unexpectedly, it's likely an error
        st.session_state.training_active = False
        # Don't assume success if thread died without logs
        if st.session_state.training_thread is not None and not st.session_state.training_thread.is_alive():
            st.warning(f"⚠️ {training_type} thread finished unexpectedly. Check logs for details.")
        else:
            st.success(
                f"✅ {completed_msg} Total time: {format_elapsed_time(total_time)}")


def render_active_training_ui(training_type: str = "Training") -> None:
    """Render UI for active training with enhanced visuals.

    Args:
        training_type: Type of training ("Training" or "Fine-Tuning") for text customization
    """
    import time

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
            progress_header = f"📊 {training_type} Progress"
            st.header(progress_header)
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
        render_all_losses_graph(all_losses_data, training_type=training_type)

    if loss_data["iterations"]:
        render_eval_losses_graph(loss_data)
        caption_text = f"💡 Page auto-refreshes every 2 seconds while {training_type.lower()}."
        st.caption(caption_text)
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
        logs_header = f"📝 {training_type} Logs (Console Output)"
        st.header(logs_header)

        # Check if there's an error in the logs
        if training_type == "Fine-Tuning":
            has_error = any(
                "Error during fine-tuning" in log or "ERROR DETECTED" in log for log in training_logs)
        else:
            has_error = any(
                "Error" in log or "ERROR" in log for log in training_logs)

        with st.expander("View All Logs", expanded=has_error):
            log_text = "\n".join(training_logs)
            st.text_area("Logs", value=log_text, height=400,
                         label_visibility="collapsed", disabled=True)
        st.caption(f"Showing {len(training_logs)} log entries")

        # If there's an error, show it prominently
        if has_error:
            st.error(
                "⚠️ **Error detected in logs above. Please scroll up to see the full error message and traceback.**")


def display_training_status(training_type: str = "Training") -> None:
    """Display training status and visualizations.

    Args:
        training_type: Type of training ("Training" or "Fine-Tuning") for text customization
    """
    # Check training status
    if st.session_state.training_thread is not None:
        thread_alive = st.session_state.training_thread.is_alive()
        training_flag_active = True
        if "training_active_flag" in st.session_state:
            with st.session_state.training_lock:
                training_flag_active = st.session_state.training_active_flag[0]

        if not thread_alive and st.session_state.training_active:
            handle_training_completion(
                training_flag_active, training_type=training_type)

    if st.session_state.training_active:
        render_active_training_ui(training_type=training_type)
    else:
        render_completed_training_ui(training_type=training_type)


def render_interactive_dashboard(
    trainer,
    metrics: Dict,
    current_step: int,
    start_time: float,
    loss_data: Dict,
    tokenizer,
    logs: List[Dict],
    title: str = "Interactive Training"
) -> None:
    """
    Render the complete interactive training dashboard (Metrics, Graphs, Analysis).
    Unified component for both Pre-Training and Fine-Tuning.
    """
    import pandas as pd
    
    max_steps = trainer.max_iters
    progress = min(current_step / max_steps, 1.0)
    
    # 1. Metrics
    latest_val_loss = None
    if loss_data["val_losses"]:
        latest_val_loss = loss_data["val_losses"][-1]
        
    render_training_metrics(
        current_iter=current_step,
        current_loss=metrics["loss"],
        running_loss=metrics["running_loss"],
        val_loss=latest_val_loss,
        progress=progress,
        max_iters=max_steps
    )
    
    # 2. Graphs
    # Training Loss
    all_losses_data = {
        "iterations": list(range(1, current_step + 1)),
        "current_losses": [m["loss"] for m in logs],
        "running_losses": [m["running_loss"] for m in logs]
    }
    render_all_losses_graph(all_losses_data, training_type=title)
    
    # Eval Loss
    if loss_data["iterations"]:
        render_eval_losses_graph(loss_data)
        
    # 3. Text & Analysis
    if "inputs" in metrics and "targets" in metrics:
        st.markdown("### 🔍 Inspect Batch")
        current_bs = metrics["inputs"].shape[0]
        # Allow user to pick which sample in the batch to view
        # Use a unique key based on step to prevent state issues
        sample_idx = st.slider(
            "Select Sample", 
            min_value=1, 
            max_value=current_bs, 
            value=1,
            key=f"sample_slider_{current_step}",
            help="Select which sequence from the current batch to inspect."
        ) - 1
        
        input_ids = metrics["inputs"][sample_idx]
        target_ids = metrics["targets"][sample_idx]
        masks = metrics["masks"][sample_idx] if "masks" in metrics else None
        
        # Determine effective length (hide padding for visual clarity)
        effective_len = None
        if masks is not None:
             try:
                 # Find last index where mask is 1 (end of response)
                 # Note: masks is boolean-like (1 for response, 0 for prompt/padding)
                 # Typically prompt is 0, response is 1, padding is 0. 
                 # We want the LAST 1 (end of valid response).
                 last_response_idx = (masks == 1).nonzero(as_tuple=True)[0][-1].item()
                 effective_len = last_response_idx + 1
             except (IndexError, RuntimeError):
                 # Fallback if no 1s found or other error
                 effective_len = len(masks)
        
        # Token Analysis
        render_token_analysis_ui(
            input_ids=input_ids[:effective_len] if effective_len else input_ids,
            target_ids=target_ids[:effective_len] if effective_len else target_ids,
            tokenizer=tokenizer,
            model=trainer.model,
            masks=masks[:effective_len] if effective_len and masks is not None else masks,
            sample_idx=sample_idx,
            n_ctx=None # Let component determine length from input
        )
        
        # Attention Heatmap
        st.divider()
        with st.expander("🔥 Attention Heatmaps", expanded=True):
             col_l, col_h = st.columns(2)
             cfg = trainer.model.cfg
             with col_l:
                 layer_idx = st.slider("Layer", 0, cfg.n_layers - 1, 0, key=f"attn_l_{current_step}_shared")
             with col_h:
                 head_idx = st.slider("Head", 0, cfg.n_heads - 1, 0, key=f"attn_h_{current_step}_shared")
             
             # We need to run the model to get attention patterns (diagnostics)
             # The trainer step might not save diagnostics to save memory.
             # So we re-run inference on this single sample.
             with st.spinner("Calculating attention..."):
                 device = get_device()
                 sample_tokens = input_ids.unsqueeze(0).to(device)
                 with torch.no_grad():
                     outputs = trainer.model(sample_tokens, return_diagnostics=True)
                     diagnostics = outputs[-1] if isinstance(outputs, tuple) else None
             
             if diagnostics and "attention_patterns" in diagnostics:
                 attn_map = diagnostics["attention_patterns"][layer_idx][0, head_idx].cpu().numpy()
                 
                 # Generate labels
                 input_ids_list = input_ids.tolist()
                 token_labels = []
                 for tid in input_ids_list:
                     try:
                         decoded = tokenizer.decode([tid])
                         token_labels.append(f"'{decoded}'")
                     except:
                         token_labels.append(f"T{tid}")
                 
                 render_attention_heatmap(attn_map, token_labels, layer_idx, head_idx)



def render_attention_heatmap(attn_map, token_labels, layer_idx, head_idx) -> None:
    """Render attention heatmap using Plotly.

    Args:
        attn_map: Attention pattern matrix [seq_len, seq_len] (numpy array)
        token_labels: List of token strings corresponding to the sequence
        layer_idx: Layer index for title
        head_idx: Head index for title
    """
    import plotly.express as px

    # Use indices for x/y to avoid collapsing duplicate tokens
    indices = list(range(len(token_labels)))

    fig = px.imshow(
        attn_map,
        x=indices,
        y=indices,
        labels=dict(x="Key (Source)", y="Query (Destination)", color="Attention"),
        title=f"Layer {layer_idx} Head {head_idx} Attention",
        color_continuous_scale="Viridis"
    )

    # Manually set tick labels to show tokens
    # If too many tokens, we might want to let plotly handle auto-ticks but usually we want to see them all
    # for detailed inspection.
    fig.update_xaxes(
        tickmode='array',
        tickvals=indices,
        ticktext=token_labels,
        tickangle=90 if len(token_labels) > 20 else 0
    )
    fig.update_yaxes(
        tickmode='array',
        tickvals=indices,
        ticktext=token_labels
    )
    fig.update_layout(height=600)  # Auto width
    st.plotly_chart(fig, width='stretch')


def render_colored_tokens_rainbow(input_ids, tokenizer) -> str:
    """
    Render tokens with rainbow coloring (for Pre-Training/Playground).
    Returns: HTML string
    """
    import html
    
    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids)
        
    input_ids_list = input_ids.tolist()
    html_output = ""
    
    # Palette of translucent colors for dark mode
    rainbow_palette = [
        "rgba(255, 107, 107, 0.4)",   # Red
        "rgba(78, 205, 196, 0.4)",    # Teal
        "rgba(255, 217, 61, 0.4)",    # Yellow
        "rgba(167, 139, 250, 0.4)",   # Purple
        "rgba(255, 159, 26, 0.4)",    # Orange
        "rgba(69, 170, 242, 0.4)",    # Blue
    ]
    
    for i, token_id in enumerate(input_ids_list):
        # Decode individual token
        try:
             token_text = tokenizer.decode([token_id])
        except:
             token_text = f"<{token_id}>"
             
        # Handle special characters for HTML
        safe_token = html.escape(token_text)
        
        # Rainbow Coloring
        bg_color = rainbow_palette[i % len(rainbow_palette)]
        title_text = f"ID: {token_id}"
        style = f"background-color: {bg_color}; border-radius: 5px; padding: 0 1px;"
        html_output += f'<span style="{style}" title="{title_text}">{safe_token}</span>'
        
    return html_output


def render_token_analysis_ui(
    input_ids,
    target_ids,
    tokenizer,
    model=None,
    masks=None,
    sample_idx=0,
    n_ctx=None
) -> None:
    """
    Render interactive token analysis UI (Input, Target, Logits).
    Shared between Pre-Training and Fine-Tuning.
    """
    import html

    # Ensure tensors
    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids)
    if not torch.is_tensor(target_ids):
        target_ids = torch.tensor(target_ids)
    if masks is not None and not torch.is_tensor(masks):
        masks = torch.tensor(masks)
        
    input_ids_list = input_ids.tolist()
    
    # 1. Generate Weighted/Colored HTML for Tokens
    
    # Check if we are in SFT mode (masks present)
    is_sft = masks is not None
    
    colored_html = "" # For Pre-Training (Rainbow)
    prompt_html = ""  # For SFT
    response_html = "" # For SFT
    
    # Palette of translucent colors for dark mode (Rainbow for Pre-Training)
    rainbow_palette = [
        "rgba(255, 107, 107, 0.4)",   # Red
        "rgba(78, 205, 196, 0.4)",    # Teal
        "rgba(255, 217, 61, 0.4)",    # Yellow
        "rgba(167, 139, 250, 0.4)",   # Purple
        "rgba(255, 159, 26, 0.4)",    # Orange
        "rgba(69, 170, 242, 0.4)",    # Blue
    ]
    
    # Pre-calculate masks info if available
    masks_list = masks.tolist() if is_sft else None
    
    for i, token_id in enumerate(input_ids_list):
        # Decode individual token
        try:
             token_text = tokenizer.decode([token_id])
        except:
             token_text = f"<{token_id}>"
             
        # Handle special characters for HTML
        safe_token = html.escape(token_text)
        
        # Determine Color & Style
        token_html_snippet = ""
        
        if is_sft:
            # SFT Mode: Prompt vs Response
            # Mask interpretation:
            # masks[i] == 1 means we compute loss on the prediction generated FROM input[i].
            # This prediction supervised input[i+1].
            # So input[i] is part of the Response (Target) ONLY IF it was predicted from the previous token.
            # i.e., masks[i-1] == 1.
            
            if i > 0 and masks_list[i-1] == 1:
                is_response_token = True
            else:
                is_response_token = False

            if is_response_token:
                bg_color = "rgba(78, 205, 196, 0.4)" # Teal (Target)
                border = "1px solid #4CAF50" # Green
                status = "Response"
                style = f"background-color: {bg_color}; border: {border}; margin: 0 1px; padding: 0 1px;"
                title_text = f"{status} | ID: {token_id}"
                response_html += f'<span style="{style}" title="{title_text}">{safe_token}</span>'
            else:
                bg_color = "rgba(255, 255, 255, 0.1)" # Grey
                border = "1px solid #555"
                status = "Prompt"
                style = f"background-color: {bg_color}; border: {border}; margin: 0 1px; padding: 0 1px;"
                title_text = f"{status} | ID: {token_id}"
                prompt_html += f'<span style="{style}" title="{title_text}">{safe_token}</span>'
        else:
            # Pre-Training Mode: Rainbow
            colored_html = render_colored_tokens_rainbow(input_ids, tokenizer)

    # Format Target Display
    # For SFT, the "Target" is the Response HTML generated above.
    # For Pre-Training, it's the next token.
    try:
        if len(target_ids) > 0:
            last_token_id = target_ids[-1].item()
            last_token_text = tokenizer.decode([last_token_id])
            target_display = f"{last_token_text}"
        else:
            target_display = "(No Target)"
            last_token_id = None
    except:
        if len(target_ids) > 0:
            target_display = f"<{target_ids[-1].item()}>"
            last_token_id = target_ids[-1].item()
        else:
            target_display = "N/A"
            last_token_id = 0

    # === LOGIT LENS (Pre-Training Only) ===
    diagnostics = None
    logits = None
    
    # We only compute/show logit lens for Pre-Training (masks is None)
    # The user explicitly requested to remove continuation prediction for SFT.
    if model is not None and not is_sft:
        # Calculate diagnostics
        with st.spinner("Calculating predictions..."):
            device = get_device()
            # Ensure input is [1, seq_len]
            if input_ids.dim() == 1:
                sample_tokens = input_ids.unsqueeze(0).to(device)
            else:
                sample_tokens = input_ids.to(device)
            
            with torch.no_grad():
                try:
                    outputs = model(sample_tokens, return_diagnostics=True)
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                    else:
                        logits = outputs
                except Exception as e:
                    # Fallback
                    try:
                        logits = model(sample_tokens)
                    except Exception as e2:
                        st.error(f"Error running model: {e2}")

    # Layout: Input (Left) | Target + Analysis (Right)
    st.markdown(f"##### 📖 Current Batch Sample ({sample_idx + 1})")
    
    c1, c2 = st.columns([3, 1])
    with c1:
        if is_sft:
             st.markdown(f"**Input (Prompt)**")
             st.markdown(
                 f'<div style="background-color: #262730; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; font-size: 14px; line-height: 1.8;">{prompt_html}</div>',
                 unsafe_allow_html=True
             )
        else:
             st.markdown(f"**Input (Context)**")
             st.markdown(
                 f'<div style="background-color: #262730; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; font-size: 14px; line-height: 1.8;">{colored_html}</div>',
                 unsafe_allow_html=True
             )
             
    with c2:
        if is_sft:
            st.markdown("**Target (Response)**")
            st.markdown(
                 f'<div style="background-color: #262730; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; font-size: 14px; line-height: 1.8; border: 1px solid #4CAF50;">{response_html}</div>',
                 unsafe_allow_html=True
            )
            # No Logit Lens for SFT
        else:
            st.markdown("**Target (Next Token)**")
            st.markdown(
                f'<div style="background-color: #262730; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; font-size: 14px; border: 1px solid #4CAF50;">{target_display}</div>',
                unsafe_allow_html=True
            )
        
            # --- LOGIT LENS UI (Only for Pre-Training) ---
            # Show predictions for what comes AFTER the displayed sequence
            if logits is not None and last_token_id is not None:
                # Get predictions for the last position
                next_token_logits = logits[0, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Get Top-5
                top_k = 5
                top_probs, top_indices = torch.topk(probs, k=top_k)
                
                # Get Actual Target Rank (Only meaningful if we know what continues)
                target_rank = "N/A"
                target_prob = 0.0
                
                try:
                    if last_token_id < probs.shape[-1]:
                        target_prob = probs[last_token_id].item()
                        target_rank = (probs > target_prob).sum().item() + 1
                except:
                    pass
                
                st.divider()
                
                st.markdown(f"**Actual Rank**: #{target_rank}")
                st.markdown(f"**Prob**: {target_prob:.2%}")
                st.caption("Next token predictions:")
                
                for i in range(top_k):
                    idx = top_indices[i].item()
                    prob = top_probs[i].item()
                    try:
                        p_token_text = tokenizer.decode([idx])
                    except:
                        p_token_text = f"<{idx}>"
                        
                    safe_p_token = html.escape(f"'{p_token_text}'")
                    
                    bar_color = "#4CAF50" if idx == last_token_id else "#262730"
                    st.markdown(
                        f"""
                        <div style="font-size: 12px; margin-bottom: 4px;">
                            <div style="display: flex; justify-content: space-between;">
                                <code>{safe_p_token}</code>
                                <span>{prob:.2%}</span>
                            </div>
                            <div style="background-color: #444; height: 4px; border-radius: 2px; width: 100%;">
                                <div style="background-color: {bar_color}; width: {prob*100}%; height: 100%; border-radius: 2px;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


# ============================================================================
# Model Configuration
# ============================================================================

# Model size presets
MODEL_SIZE_PRESETS = {
    "small": {
        "d_model": 256,
        "n_heads": 4,
        "n_layers": 4,
        "n_ctx": 256,
        "d_head": 64,
        "d_mlp": 1024,
    },
    "medium": {
        "d_model": 512,
        "n_heads": 8,
        "n_layers": 6,
        "n_ctx": 512,
        "d_head": 64,
        "d_mlp": 2048,
    },
    "full": {
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 12,
        "n_ctx": 1024,
        "d_head": 64,
        "d_mlp": 3072,
    },
}


def apply_model_size_preset(size: str, config: Dict) -> None:
    """Apply model size preset to config.

    Note: This function does NOT set n_kv_heads - that should be set by
    architecture presets (which know whether to use MHA, GQA, or MQA).
    """
    preset = MODEL_SIZE_PRESETS[size]
    for key, value in preset.items():
        config[key] = value


def apply_architecture_preset(preset_name: str, config: Dict) -> None:
    """Apply architecture preset (GPT, LLaMA 4, OLMo 3) to config."""
    if preset_name == "GPT":
        config["positional_encoding"] = "learned"
        config["normalization"] = "layernorm"
        config["activation"] = "gelu"
        config["tokenizer_type"] = "bpe-tiktoken"
        config["use_moe"] = False
        # GPT uses MHA (n_kv_heads = n_heads)
        config["n_kv_heads"] = config.get("n_heads", 4)
    elif preset_name == "LLAMA":
        config["positional_encoding"] = "rope"
        config["normalization"] = "rmsnorm"
        config["activation"] = "swiglu"
        config["tokenizer_type"] = "sentencepiece"
        config["rope_theta"] = 10000.0
        # LLaMA 4 uses MoE architecture (first LLaMA to use MoE)
        # Scout: 16 experts, Maverick: 128 experts (defaulting to Scout)
        config["use_moe"] = True
        config["num_experts"] = 16  # Scout configuration (Maverick uses 128)
        # Each token goes to 1 routed expert + shared expert
        config["num_experts_per_tok"] = 1
        config["use_shared_experts"] = True  # LLaMA 4 uses shared experts
        config["num_shared_experts"] = 1
        config["router_type"] = "top_k_with_shared"
        config["load_balancing_loss_weight"] = 0.01
        config["expert_capacity_factor"] = 1.25
        # LLaMA 4 likely uses GQA (modern standard, 4:1 ratio similar to Mixtral/Mistral)
        n_heads = config.get("n_heads", 32)
        config["n_kv_heads"] = max(1, n_heads // 4)
    elif preset_name == "OLMO":
        config["positional_encoding"] = "alibi"
        config["normalization"] = "layernorm"
        config["activation"] = "swiglu"
        config["tokenizer_type"] = "sentencepiece"
        config["use_moe"] = False
        # OLMo 3 uses MHA (n_kv_heads = n_heads) - dense model architecture
        config["n_kv_heads"] = config.get("n_heads", 4)


def apply_deepseek_v2_preset(config: Dict) -> None:
    """Apply DeepSeek V2 preset with MoE."""
    config["positional_encoding"] = "rope"
    config["normalization"] = "rmsnorm"
    config["activation"] = "swiglu"
    config["tokenizer_type"] = "sentencepiece"
    config["rope_theta"] = 10000.0
    config["use_moe"] = True
    config["num_experts"] = 64
    config["num_experts_per_tok"] = 6
    config["use_shared_experts"] = True
    config["num_shared_experts"] = 2
    config["router_type"] = "top_k_with_shared"
    config["load_balancing_loss_weight"] = 0.01
    config["expert_capacity_factor"] = 1.25
    # DeepSeek V2 uses MHA (n_kv_heads = n_heads)
    config["n_kv_heads"] = config.get("n_heads", 4)


def apply_mixtral_preset(config: Dict) -> None:
    """Apply Mixtral 8x7B preset.

    Mixtral uses a Sparse Mixture-of-Experts (MoE) architecture:
    - 8 experts per MoE layer
    - Top-2 routing (activates 2 experts per token)
    - Based on LLaMA architecture (RoPE, RMSNorm, SwiGLU)
    - Uses Grouped Query Attention (GQA) - typically 32 Q heads, 8 KV heads (4:1 ratio)
    - No shared experts (standard top-k routing)
    - Total: 46.7B parameters, but only ~12.9B active per token
    """
    config["positional_encoding"] = "rope"
    config["normalization"] = "rmsnorm"
    config["activation"] = "swiglu"
    config["tokenizer_type"] = "sentencepiece"
    config["rope_theta"] = 10000.0
    config["use_moe"] = True
    config["num_experts"] = 8
    config["num_experts_per_tok"] = 2
    config["use_shared_experts"] = False
    config["num_shared_experts"] = 0
    config["router_type"] = "top_k"
    config["load_balancing_loss_weight"] = 0.01
    config["expert_capacity_factor"] = 1.25
    # Set GQA: use 4:1 ratio (e.g., 32 Q heads -> 8 KV heads, or 8 Q heads -> 2 KV heads)
    n_heads = config.get("n_heads", 32)
    # GQA with 4:1 ratio (Mixtral-style)
    config["n_kv_heads"] = max(1, n_heads // 4)


def apply_llama_moe_preset(config: Dict) -> None:
    """Deprecated: Use apply_mixtral_preset instead. Kept for backward compatibility."""
    apply_mixtral_preset(config)


def render_model_config_ui() -> Dict:
    """Render model configuration UI and return config dict."""
    # Initialize config if needed
    if "model_config" not in st.session_state:
        st.session_state.model_config = _get_default_config()

    config = st.session_state.model_config

    # Preset buttons
    _render_preset_buttons(config)

    # Model size selector (right after presets)
    _render_model_size_selector(config)

    # Model components
    _render_model_components(config)

    # Model dimensions
    _render_model_dimensions(config)

    # RoPE settings (conditional)
    if config["positional_encoding"] == "rope":
        _render_rope_settings(config)

    # MoE settings (conditional)
    _render_moe_settings(config)

    return config


def _get_default_config() -> Dict:
    """Get default model configuration."""
    return {
        "positional_encoding": "learned",
        "normalization": "layernorm",
        "activation": "gelu",
        "model_size": "small",
        "d_model": 256,
        "n_heads": 4,
        "n_kv_heads": 4,  # Default to MHA (n_kv_heads = n_heads)
        "n_layers": 4,
        "n_ctx": 256,
        "d_head": 64,
        "d_mlp": 1024,
        "rope_theta": 10000.0,
        "tokenizer_type": "bpe-tiktoken",
        "use_moe": False,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "use_shared_experts": False,
        "num_shared_experts": 2,
        "router_type": "top_k",
        "load_balancing_loss_weight": 0.01,
        "expert_capacity_factor": 1.25,
    }


def _render_preset_buttons(config: Dict) -> None:
    """Render architecture preset buttons."""
    st.subheader("Quick Presets")
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1.5])

    with col1:
        if st.button("🚀 GPT-2", width='stretch'):
            apply_architecture_preset("GPT", config)
            apply_model_size_preset(config.get("model_size", "small"), config)
            # Ensure MHA after size preset (size preset might have changed n_heads)
            config["n_kv_heads"] = config.get("n_heads", 4)
            st.rerun()

    with col2:
        if st.button("🦙 LLaMA 4", width='stretch'):
            apply_architecture_preset("LLAMA", config)
            apply_model_size_preset(config.get("model_size", "small"), config)
            # Re-apply LLaMA 4's GQA setting after size preset
            n_heads = config.get("n_heads", 4)
            config["n_kv_heads"] = max(1, n_heads // 4)  # GQA with 4:1 ratio
            st.rerun()

    with col3:
        if st.button("🔬 OLMo 3", width='stretch'):
            apply_architecture_preset("OLMO", config)
            apply_model_size_preset(config.get("model_size", "small"), config)
            # Ensure MHA after size preset (size preset might have changed n_heads)
            config["n_kv_heads"] = config.get("n_heads", 4)
            st.rerun()

    with col4:
        if st.button("🔷 DeepSeek V2", width='stretch'):
            apply_deepseek_v2_preset(config)
            apply_model_size_preset(config.get("model_size", "small"), config)
            # Ensure MHA after size preset (size preset might have changed n_heads)
            config["n_kv_heads"] = config.get("n_heads", 4)
            st.rerun()

    with col5:
        if st.button("🎯 Mixtral", width='stretch'):
            apply_mixtral_preset(config)
            apply_model_size_preset(config.get("model_size", "small"), config)
            # Re-apply Mixtral's GQA setting after size preset (size preset changes n_heads)
            n_heads = config.get("n_heads", 4)
            config["n_kv_heads"] = max(1, n_heads // 4)  # GQA with 4:1 ratio
            st.rerun()

    st.markdown("---")
    with st.expander("ℹ️ About Presets", expanded=False):
        st.markdown(_get_preset_info())


def _render_model_components(config: Dict) -> None:
    """Render model component selectors."""
    st.subheader("Model Components")
    col1, col2, col3 = st.columns(3)

    with col1:
        config["positional_encoding"] = st.selectbox(
            "Positional Encoding",
            ["learned", "rope", "alibi", "none"],
            index=["learned", "rope", "alibi", "none"].index(
                config["positional_encoding"]
            ),
            help="Learned: GPT-style embeddings\nRoPE: Rotary Position Embedding (LLaMA)\nALiBi: Attention with Linear Biases (OLMo)\nNone: No positional encoding"
        )

    with col2:
        config["normalization"] = st.selectbox(
            "Normalization",
            ["layernorm", "rmsnorm"],
            index=["layernorm", "rmsnorm"].index(config["normalization"]),
            help="LayerNorm: GPT/OLMo style\nRMSNorm: LLaMA style (simpler, faster)"
        )

    with col3:
        config["activation"] = st.selectbox(
            "Activation Function",
            ["gelu", "swiglu"],
            index=["gelu", "swiglu"].index(config["activation"]),
            help="GELU: GPT style\nSwiGLU: LLaMA/OLMo style (gated)"
        )

    # Attention type selector (GQA/MQA support)
    st.markdown("---")
    attention_type_options = [
        "Multi-Head (MHA)", "Grouped Query (GQA)", "Multi-Query (MQA)"]
    current_n_kv_heads = config.get("n_kv_heads", config.get("n_heads", 12))
    current_n_heads = config.get("n_heads", 12)

    # Determine current attention type
    if current_n_kv_heads == current_n_heads:
        current_attention_type = "Multi-Head (MHA)"
    elif current_n_kv_heads == 1:
        current_attention_type = "Multi-Query (MQA)"
    else:
        current_attention_type = "Grouped Query (GQA)"

    attention_type_index = attention_type_options.index(
        current_attention_type) if current_attention_type in attention_type_options else 0

    attention_type = st.selectbox(
        "Attention Type",
        attention_type_options,
        index=attention_type_index,
        help="MHA: Standard multi-head attention (each head has separate Q/K/V)\n"
        "GQA: Grouped Query Attention - groups of Q heads share K/V heads (efficient, used in LLaMA 2, Mistral)\n"
        "MQA: Multi-Query Attention - all Q heads share single K/V head (most efficient, slight quality trade-off)"
    )

    # Update n_kv_heads based on selection
    if attention_type == "Multi-Head (MHA)":
        config["n_kv_heads"] = current_n_heads
    elif attention_type == "Multi-Query (MQA)":
        config["n_kv_heads"] = 1
    else:  # GQA
        # Show n_kv_heads input for GQA
        if "n_kv_heads" not in config or config["n_kv_heads"] == current_n_heads or config["n_kv_heads"] == 1:
            # Set a reasonable default for GQA (e.g., 8 KV heads for 32 Q heads)
            default_kv_heads = max(1, current_n_heads //
                                   4) if current_n_heads >= 4 else 1
            config["n_kv_heads"] = default_kv_heads

        col_gqa1, col_gqa2 = st.columns([1, 2])
        with col_gqa1:
            config["n_kv_heads"] = st.number_input(
                "n_kv_heads (KV Heads)",
                min_value=1,
                max_value=current_n_heads,
                value=config["n_kv_heads"],
                step=1,
                help=f"Number of KV heads (must divide {current_n_heads}). "
                f"Lower values = smaller KV cache, faster inference. "
                f"Common: {current_n_heads // 4} or {current_n_heads // 2} for GQA."
            )

        with col_gqa2:
            if current_n_heads % config["n_kv_heads"] != 0:
                st.error(
                    f"n_kv_heads ({config['n_kv_heads']}) must divide n_heads ({current_n_heads})")
            elif config["n_kv_heads"] == current_n_heads:
                st.info("This is equivalent to Multi-Head Attention (MHA)")
            elif config["n_kv_heads"] == 1:
                st.info("This is equivalent to Multi-Query Attention (MQA)")
            else:
                kv_cache_reduction = (
                    1 - config["n_kv_heads"] / current_n_heads) * 100
                st.success(
                    f"KV cache reduced by {kv_cache_reduction:.1f}% vs MHA")


def _render_model_dimensions(config: Dict) -> None:
    """Render model dimension inputs."""
    st.subheader("Model Dimensions")
    col1, col2, col3 = st.columns(3)

    with col1:
        config["d_model"] = st.number_input(
            "d_model (Model Dimension)",
            min_value=64, max_value=4096, value=config["d_model"], step=64,
            help="Hidden dimension size"
        )
        config["n_heads"] = st.number_input(
            "n_heads (Number of Heads)",
            min_value=1, max_value=64, value=config["n_heads"],
            help="Number of attention heads"
        )

    with col2:
        config["n_layers"] = st.number_input(
            "n_layers (Number of Layers)",
            min_value=1, max_value=128, value=config["n_layers"],
            help="Number of transformer layers"
        )
        config["n_ctx"] = st.number_input(
            "n_ctx (Context Length)",
            min_value=64, max_value=8192, value=config["n_ctx"], step=64,
            help="Maximum sequence length"
        )

    with col3:
        config["d_head"] = st.number_input(
            "d_head (Head Dimension)",
            min_value=32, max_value=256, value=config["d_head"], step=32,
            help="Dimension per attention head"
        )
        config["d_mlp"] = st.number_input(
            "d_mlp (MLP Dimension)",
            min_value=128, max_value=16384, value=config["d_mlp"], step=128,
            help="MLP hidden dimension (typically 4x d_model)"
        )


def _render_model_size_selector(config: Dict) -> None:
    """Render model size selector as buttons."""
    st.subheader("Model Size Preset")
    current_size = config.get("model_size", "small")

    col1, col2, col3 = st.columns(3)

    with col1:
        button_type = "primary" if current_size == "small" else "secondary"
        if st.button("🔹 Small", width='stretch', type=button_type,
                     help="Small model: 256 dim, 4 heads, 4 layers"):
            if current_size != "small":
                config["model_size"] = "small"
                apply_model_size_preset("small", config)
                st.rerun()

    with col2:
        button_type = "primary" if current_size == "medium" else "secondary"
        if st.button("🔸 Medium", width='stretch', type=button_type,
                     help="Medium model: 512 dim, 8 heads, 6 layers"):
            if current_size != "medium":
                config["model_size"] = "medium"
                apply_model_size_preset("medium", config)
                st.rerun()

    with col3:
        button_type = "primary" if current_size == "full" else "secondary"
        if st.button("🔶 Full", width='stretch', type=button_type,
                     help="Full model: 768 dim, 12 heads, 12 layers"):
            if current_size != "full":
                config["model_size"] = "full"
                apply_model_size_preset("full", config)
                st.rerun()

    config["model_size"] = current_size


def _render_rope_settings(config: Dict) -> None:
    """Render RoPE-specific settings."""
    config["rope_theta"] = st.number_input(
        "RoPE Theta (Base Frequency)",
        min_value=1000.0, max_value=1000000.0,
        value=config["rope_theta"], step=1000.0, format="%.0f",
        help="Base frequency for RoPE. LLaMA 1/2: 10000, LLaMA 3: 500000"
    )


def _render_moe_settings(config: Dict) -> None:
    """Render MoE (Mixture of Experts) configuration settings."""
    st.subheader("Mixture of Experts (MoE)")

    config["use_moe"] = st.checkbox(
        "Enable MoE",
        value=config.get("use_moe", False),
        help="Enable Mixture of Experts: use multiple expert MLPs with routing"
    )

    if config["use_moe"]:
        with st.expander("MoE Configuration", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                config["num_experts"] = st.slider(
                    "Number of Experts",
                    min_value=2,
                    max_value=64,
                    value=config.get("num_experts", 8),
                    step=1,
                    help="Total number of expert MLPs"
                )
                config["num_experts_per_tok"] = st.slider(
                    "Experts per Token (Top-k)",
                    min_value=1,
                    max_value=8,
                    value=config.get("num_experts_per_tok", 2),
                    step=1,
                    help="Number of experts to activate per token (top-k routing)"
                )
                config["router_type"] = st.selectbox(
                    "Router Type",
                    ["top_k", "top_k_with_shared"],
                    index=0 if config.get(
                        "router_type", "top_k") == "top_k" else 1,
                    help="Top-k: Standard routing\nTop-k with Shared: Some experts always active (DeepSeek-style)"
                )

            with col2:
                config["use_shared_experts"] = st.checkbox(
                    "Use Shared Experts",
                    value=config.get("use_shared_experts", False),
                    help="Enable shared experts that are always active (DeepSeek-style)"
                )
                if config["use_shared_experts"]:
                    config["num_shared_experts"] = st.slider(
                        "Number of Shared Experts",
                        min_value=1,
                        max_value=8,
                        value=config.get("num_shared_experts", 2),
                        step=1,
                        help="Number of always-active shared experts"
                    )
                config["load_balancing_loss_weight"] = st.number_input(
                    "Load Balancing Loss Weight",
                    min_value=0.0,
                    max_value=0.1,
                    value=config.get("load_balancing_loss_weight", 0.01),
                    step=0.001,
                    format="%.3f",
                    help="Weight for load balancing auxiliary loss (encourages uniform expert usage)"
                )
                config["expert_capacity_factor"] = st.number_input(
                    "Expert Capacity Factor",
                    min_value=1.0,
                    max_value=2.0,
                    value=config.get("expert_capacity_factor", 1.25),
                    step=0.05,
                    format="%.2f",
                    help="Capacity factor for expert load balancing (higher = more capacity per expert)"
                )

        # Update router_type based on use_shared_experts if needed
        if config["use_shared_experts"]:
            config["router_type"] = "top_k_with_shared"
        elif config.get("router_type") == "top_k_with_shared" and not config["use_shared_experts"]:
            config["router_type"] = "top_k"


def _get_preset_info() -> str:
    """Get preset information markdown."""
    return """
    **Preset Configurations:**
    - **GPT-2**: Learned positional embeddings, LayerNorm, GELU activation, BPE-tiktoken (GPT-2 style)
    - **LLaMA 4**: RoPE positional encoding, RMSNorm, SwiGLU activation, SentencePiece tokenizer, MoE architecture (16 experts, shared expert), GQA attention (4:1 ratio)
    - **OLMo 3**: ALiBi positional encoding, LayerNorm, SwiGLU activation, SentencePiece tokenizer, dense model (no MoE), MHA attention
    - **DeepSeek V2**: LLaMA-style with MoE (64 experts, top-6, 2 shared experts)
    - **Mixtral**: LLaMA-style with MoE (8 experts, top-2 routing). Sparse MoE architecture with 8 experts per layer, activating 2 experts per token. Based on Mixtral 8x7B design.

    **Model Size:**
    - Controls model dimensions (d_model, n_heads, n_layers, etc.)
    - All presets use the same dimensions for each size
    - Clicking a preset uses the currently selected model size

    **MoE (Mixture of Experts):**
    - MoE models use multiple expert MLPs with routing
    - Only a subset of experts are activated per token (more efficient)
    - **DeepSeek V2**: Uses shared experts (always active) + routed experts (top-k with shared)
    - **Mixtral**: Uses standard top-k routing (8 experts, top-2 per token)

    **Customization:**
    - All options below can be manually adjusted after selecting a preset
    - Tokenizer is automatically set but can be changed
    """


def generate_graphviz_architecture(config: Dict) -> str:
    """Generate Graphviz DOT code for transformer architecture (Vertical, Beige Theme)."""
    n_layers = config.get("n_layers", 4)
    n_heads = config.get("n_heads", 4)
    pos_enc = config.get("positional_encoding", "learned")
    activation = config.get("activation", "gelu")
    
    # Theme
    COLOR_BOX_FILL = "#E6DAC3"  # Beige
    COLOR_BOX_BORDER = "#5C5248" # Dark Brown/Grey
    COLOR_TEXT = "#4A4A4A"
    FONT_MAIN = "Helvetica"
    FONT_MATH = "Times-Roman" 
    
    dot = []
    dot.append('digraph TransformerArchitecture {')
    dot.append('    bgcolor="transparent";')
    dot.append('    rankdir=BT;')
    dot.append('    nodesep=0.5;')
    dot.append('    ranksep=0.5;')
    # splines=ortho is good for boxes, but sometimes makes the main line janky. 
    # splines=line or polyline might be safer for the main vertical, but ortho is requested style.
    # We will trust 'group' to keep it straight.
    dot.append('    splines=ortho;') 
    
    # Global styles
    dot.append(f'    node [shape=rect, style="filled,rounded", fillcolor="{COLOR_BOX_FILL}", color="{COLOR_BOX_BORDER}", penwidth=1.5, fontname="{FONT_MAIN}", fontcolor="{COLOR_TEXT}"];')
    dot.append(f'    edge [color="{COLOR_BOX_BORDER}", penwidth=1.2, arrowsize=0.8];')
    
    # Helper for Note Labels
    def make_note_label(text, subtext=""):
        return f'''<
        <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="12" FACE="{FONT_MAIN}">{text}</FONT></TD></TR>
        <TR><TD ALIGN="LEFT"><FONT POINT-SIZE="14" FACE="{FONT_MATH}"><i>{subtext}</i></FONT></TD></TR>
        </TABLE>
        >'''

    # === Pre-calculated Dynamic Labels ===
    # 1. Heads Label
    n_kv = config.get("n_kv_heads", n_heads)
    if n_kv == n_heads:
        heads_txt = "Attention Heads"
    else:
        heads_txt = f"GQA Heads ({n_kv} KV)"
    if pos_enc == "rope":
        heads_txt += "\\n(RoPE)"
    elif pos_enc == "alibi":
        heads_txt += "\\n(ALiBi)"
        
    # 2. MLP Label
    mlp_txt = "MLP m"
    if config.get("use_moe", False):
        n_exp = config.get('num_experts', 8)
        k_exp = config.get('num_experts_per_tok', 2)
        mlp_txt = f"MoE MLP\\n({n_exp} Experts, top-{k_exp})"
    elif activation == "swiglu":
        mlp_txt += "\\n(SwiGLU)"
        
    # === NODES ===
    
    # Input/Output Stages
    dot.append(f'    tokens [label="tokens", shape=box3d, group=main];')
    dot.append(f'    embed [label="embed", group=main];')
    dot.append(f'    note_embed [shape=plaintext, style=none, label={make_note_label("Token embedding.", "x&#8320; = W&#7431;t")}];')
    
    dot.append(f'    unembed [label="unembed", group=main];')
    dot.append(f'    logits [label="logits", group=main];')
    dot.append(f'    note_unembed [shape=plaintext, style=none, label={make_note_label("Unembedding", "T(t) = W&#7516; x&#8331;&#8321;")}];')

    # Connections for Input/Output
    dot.append('    tokens -> embed [weight=100];') # High weight for vertical alignment
    dot.append('    unembed -> logits [weight=100];') 
    
    # Equation Alignment
    dot.append('    { rank=same; tokens; note_embed; }')
    dot.append('    { rank=same; unembed; note_unembed; }')
    dot.append('    embed -> note_embed [style=invis, minlen=2];')
    dot.append('    unembed -> note_unembed [style=invis, minlen=2];')


    # === CLUSTER (The Residual Block) ===
    dot.append(f'    subgraph cluster_residual {{')
    dot.append('        label="One residual block (repeated)";')
    dot.append('        fontcolor="#888888";')
    dot.append('        style=dashed;')
    dot.append('        color="#888888";')
    dot.append('        margin=20;') # Add breathing room
    
    # Nodes INSIDE cluster
    dot.append(f'        x0 [shape=plaintext, style=none, label=<<i>x</i><sub>0</sub>>, group=main];')
    dot.append(f'        x1 [shape=plaintext, style=none, label=<<i>x</i><sub>i+1</sub>>, group=main];')
    dot.append(f'        x2 [shape=plaintext, style=none, label=<<i>x</i><sub>i+2</sub>>, group=main];')
    
    dot.append('        plus1 [shape=circle, label="+", fixedsize=true, width=0.4, group=main];')
    dot.append('        plus2 [shape=circle, label="+", fixedsize=true, width=0.4, group=main];')
    
    dot.append(f'        heads [label="{heads_txt}", width=2.5];')
    dot.append(f'        mlp [label="{mlp_txt}", width=2.5];')
    
    # Notes inside cluster or just aligned? 
    # To be "next to" the block, they don't strictly need to be inside the cluster border, 
    # but aligning them with nodes inside is tricky if they are outside. 
    # Let's put them inside for now to ensure alignment works, or keep valid rank=same.
    dot.append(f'        note_attn [shape=plaintext, style=none, label={make_note_label("Attention Layer", "x&#7522;&#8330;&#8321; = x&#7522; + &Sigma; h(x&#7522;)")}];')
    dot.append(f'        note_mlp [shape=plaintext, style=none, label={make_note_label("MLP Layer", "x&#7522;&#8330;&#8322; = x&#7522;&#8330;&#8321; + m(x&#7522;&#8330;&#8321;)")}];')

    # Ranks inside cluster
    dot.append('        { rank=same; x0; heads; note_attn; }')
    dot.append('        { rank=same; x1; mlp; note_mlp; }')

    # Block Connections
    # 1. Main Stream (Vertical Spine) - High weight, same group
    dot.append('        x0 -> plus1 [weight=100];') 
    dot.append('        plus1 -> x1 [weight=100];')
    dot.append('        x1 -> plus2 [weight=100];')
    dot.append('        plus2 -> x2 [weight=100];')
    
    # 2. Residual Branches - Constraint=false ensures they don't impact vertical layout ranking
    # Branch OUT
    dot.append('        x0 -> heads [constraint=false];')
    dot.append('        x1 -> mlp [constraint=false];')
    
    # Branch IN
    dot.append('        heads -> plus1;')
    dot.append('        mlp -> plus2;')
    
    # 3. Note alignment
    dot.append('        heads -> note_attn [style=invis, minlen=1];')
    dot.append('        mlp -> note_mlp [style=invis, minlen=1];')

    dot.append('    }') # End Cluster

    # === Connecting Cluster to Outer World ===
    dot.append('    embed -> x0 [weight=100];')
    
    # Link to final output
    dot.append(f'    x_final [shape=plaintext, style=none, label=<<i>x</i><sub>-1</sub>>, group=main];')
    dot.append('    x2 -> x_final [style=dotted, label="repeat...", weight=100];')
    dot.append('    x_final -> unembed [weight=100];')

    dot.append('}')
    return "\n".join(dot)



def render_model_architecture_diagram(config: Dict) -> None:
    """Render the model architecture diagram in Streamlit."""
    with st.expander("🏗️ Architecture Diagram", expanded=False):
        try:
            import graphviz
            dot_code = generate_graphviz_architecture(config)
            st.graphviz_chart(dot_code)
        except ImportError:
            st.warning(
                "Graphviz is not installed. Install it with: `pip install graphviz`")
            st.code(generate_graphviz_architecture(config), language="dot")

        # Add explanation
        st.markdown("""
        ** Legend:**
        - The **vertical residual stream** (x₀ → x₁ → x₂ → ...) carries information through the network
        - **Attention heads** and **MLP blocks** branch off and add their contributions back with "+"
        - The **dashed box** shows one residual block that repeats multiple times
        """)


def render_model_equations(config: Dict) -> None:
    """Render full mathematical equations for the model architecture."""
    with st.expander("📐 Equations", expanded=False):
        d_model = config.get("d_model", 256)
        n_heads = config.get("n_heads", 4)
        n_kv_heads = config.get("n_kv_heads", n_heads)  # Default to MHA
        d_head = config.get("d_head", 64)
        d_mlp = config.get("d_mlp", 1024)
        pos_enc = config.get("positional_encoding", "learned")
        norm = config.get("normalization", "layernorm")
        activation = config.get("activation", "gelu")
        rope_theta = config.get("rope_theta", 10000.0)
        use_moe = config.get("use_moe", False)
        num_experts = config.get("num_experts", 8)
        num_experts_per_tok = config.get("num_experts_per_tok", 2)
        use_shared_experts = config.get("use_shared_experts", False)
        num_shared_experts = config.get("num_shared_experts", 2)
        load_balancing_loss_weight = config.get(
            "load_balancing_loss_weight", 0.01)

        st.markdown("### Key Notation")
        st.markdown("""
        - **x**: Input tensor $[B, L, d_{model}]$ where $B$ = batch size, $L$ = sequence length
        - **h**: Hidden state $[B, L, d_{model}]$
        - **W_Q, W_K, W_V, W_O**: Attention weight matrices
        - **W_in, W_out**: MLP weight matrices
        - **b_in, b_out**: MLP bias vectors
        - **d_model**: Model dimension
        - **d_head**: Dimension per attention head
        - **n_heads**: Number of attention heads
        - **d_mlp**: MLP hidden dimension
        - **i, j**: Position indices
        """)

        st.markdown("---")
        st.markdown("### 1. Token Embedding")
        st.latex(r"E \in \mathbb{R}^{V \times d_{model}}")
        st.latex(
            r"x_0 = E[\text{tokens}] \quad \text{where } x_0 \in \mathbb{R}^{B \times L \times d_{model}}")
        st.markdown(
            f"where $V$ = vocabulary size, $B$ = batch size, $L$ = sequence length, $d_{{model}} = {d_model}$")

        st.markdown("---")
        st.markdown("### 2. Positional Encoding")

        if pos_enc == "learned":
            st.markdown("**Learned Positional Embeddings (GPT-style)**")
            st.latex(r"P \in \mathbb{R}^{L_{max} \times d_{model}}")
            st.latex(r"\text{pos} = P[\text{positions}]")
            st.latex(r"x_0 = x_0 + \text{pos}")
            st.markdown(
                f"where $L_{{max}}$ = maximum sequence length, $d_{{model}} = {d_model}$")
        elif pos_enc == "rope":
            st.markdown("**RoPE (Rotary Position Embedding) - LLaMA-style**")
            st.markdown("For each position $i$ and head dimension $d$:")
            st.latex(r"\theta_d = 10000^{-2d/d_{head}}")
            st.latex(
                r"R_i = \begin{bmatrix} \cos(\theta_d \cdot i) & -\sin(\theta_d \cdot i) \\ \sin(\theta_d \cdot i) & \cos(\theta_d \cdot i) \end{bmatrix}")
            st.markdown(
                "Applied to Q and K vectors during attention (see Attention section):")
            st.latex(
                r"q_{\text{rotated}} = R_i \cdot q \quad \text{(rotate query by position } i\text{)}")
            st.latex(
                r"k_{\text{rotated}} = R_j \cdot k \quad \text{(rotate key by position } j\text{)}")
            st.markdown(
                f"where $\\theta = {rope_theta}$ (base frequency), $d_{{head}} = {d_head}$")
        elif pos_enc == "alibi":
            st.markdown(
                "**ALiBi (Attention with Linear Biases) - OLMo-style**")
            st.latex(r"\text{bias}(i, j) = -m_h \cdot |i - j|")
            st.markdown("where $m_h$ is a head-specific slope:")
            st.latex(r"m_h = 2^{-8h/n_{heads}}")
            st.markdown(
                "Applied to attention scores during attention computation (see Attention section):")
            st.latex(
                r"\text{attn\_scores} = \text{attn\_scores} + \text{bias\_matrix}")
            st.markdown(f"where $n_{{heads}} = {n_heads}$")

        st.markdown("---")
        st.markdown("### 3. Transformer Block (Repeated)")
        st.markdown("""
        Each transformer block consists of:
        1. Pre-norm attention with residual connection
        2. Pre-norm MLP with residual connection
        """)

        st.markdown("#### 3.1 Attention Sub-block")

        if norm == "layernorm":
            st.markdown("**Pre-Normalization (LayerNorm):**")
            st.latex(r"\mu = \frac{1}{d_{model}} \sum_{k=1}^{d_{model}} x_k")
            st.latex(
                r"\sigma^2 = \frac{1}{d_{model}} \sum_{k=1}^{d_{model}} (x_k - \mu)^2")
            st.latex(
                r"x_{\text{norm}} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta")
            st.markdown(
                "where $\\epsilon = 10^{-5}$ (small constant), $\\gamma$ and $\\beta$ are learnable parameters")
        elif norm == "rmsnorm":
            st.markdown("**Pre-Normalization (RMSNorm):**")
            st.latex(
                r"\sigma^2 = \frac{1}{d_{model}} \sum_{k=1}^{d_{model}} x_k^2")
            st.latex(
                r"x_{\text{norm}} = \frac{x}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma")
            st.markdown(
                "where $\\epsilon = 10^{-5}$ (small constant), $\\gamma$ is a learnable scale parameter (no bias $\\beta$)")

        # Determine attention type
        if n_kv_heads == n_heads:
            attention_type = "Multi-Head Attention (MHA)"
        elif n_kv_heads == 1:
            attention_type = "Multi-Query Attention (MQA)"
        else:
            attention_type = "Grouped Query Attention (GQA)"

        st.markdown(f"**{attention_type}:**")

        if n_kv_heads == n_heads:
            # Standard MHA
            st.markdown("Project to Q, K, V for all heads:")
            st.latex(
                r"Q = x_{\text{norm}} W_Q^T \quad Q \in \mathbb{R}^{B \times L \times n_{heads} \times d_{head}}")
            st.latex(
                r"K = x_{\text{norm}} W_K^T \quad K \in \mathbb{R}^{B \times L \times n_{heads} \times d_{head}}")
            st.latex(
                r"V = x_{\text{norm}} W_V^T \quad V \in \mathbb{R}^{B \times L \times n_{heads} \times d_{head}}")
            st.markdown(
                f"where $W_Q, W_K, W_V \\in \\mathbb{{R}}^{{{n_heads} \\times {d_head} \\times {d_model}}}$")
        else:
            # GQA or MQA
            st.markdown("Project to Q, K, V:")
            st.latex(
                r"Q = x_{\text{norm}} W_Q^T \quad Q \in \mathbb{R}^{B \times L \times n_{heads} \times d_{head}}")
            st.latex(
                r"K = x_{\text{norm}} W_K^T \quad K \in \mathbb{R}^{B \times L \times n_{kv\_heads} \times d_{head}}")
            st.latex(
                r"V = x_{\text{norm}} W_V^T \quad V \in \mathbb{R}^{B \times L \times n_{kv\_heads} \times d_{head}}")
            st.markdown(
                f"where $W_Q \\in \\mathbb{{R}}^{{{n_heads} \\times {d_head} \\times {d_model}}}$, "
                f"$W_K, W_V \\in \\mathbb{{R}}^{{{n_kv_heads} \\times {d_head} \\times {d_model}}}$")

            # Show broadcasting step
            repeat_factor = n_heads // n_kv_heads
            st.markdown("**Broadcast K/V to match Q heads:**")
            st.latex(
                r"K_{{\text{{broadcast}}}} = \text{{repeat}}(K, \text{{dim}}=2, \text{{times}}={}) \quad K_{{\text{{broadcast}}}} \in \mathbb{{R}}^{{B \times L \times n_{{heads}} \times d_{{head}}}}".format(repeat_factor))
            st.latex(
                r"V_{{\text{{broadcast}}}} = \text{{repeat}}(V, \text{{dim}}=2, \text{{times}}={}) \quad V_{{\text{{broadcast}}}} \in \mathbb{{R}}^{{B \times L \times n_{{heads}} \times d_{{head}}}}".format(repeat_factor))
            st.markdown(
                f"Each KV head is repeated {repeat_factor} times to match the {n_heads} Q heads. "
                f"This allows {n_heads} Q heads to share {n_kv_heads} KV heads, reducing KV cache size by "
                f"{(1 - n_kv_heads/n_heads)*100:.1f}%.")

        if pos_enc == "rope":
            st.markdown("**Apply RoPE (Rotary Position Embedding):**")
            st.markdown(
                "RoPE rotates Q and K vectors BEFORE computing attention scores:")
            st.latex(
                r"R_i = \begin{bmatrix} \cos(\theta_d \cdot i) & -\sin(\theta_d \cdot i) \\ \sin(\theta_d \cdot i) & \cos(\theta_d \cdot i) \end{bmatrix}")
            if n_kv_heads == n_heads:
                st.latex(
                    r"Q_{\text{rotated}} = R_i \cdot Q \quad K_{\text{rotated}} = R_j \cdot K")
            else:
                st.latex(
                    r"Q_{\text{rotated}} = R_i \cdot Q \quad K_{\text{rotated}} = R_j \cdot K_{\text{original}}")
                st.markdown(
                    f"Note: RoPE is applied to K with {n_kv_heads} heads, then K is broadcasted to match Q.")
            st.markdown("Compute attention scores with rotated vectors:")
            st.latex(
                r"\text{attn\_scores} = \frac{Q_{\text{rotated}} K_{\text{rotated}}^T}{\sqrt{d_{head}}}")
            st.markdown(
                f"where $\\theta_d = 10000^{{-2d/{d_head}}}$, $\\theta = {rope_theta}$, $d_{{{'head'}}} = {d_head}$")
            st.markdown(
                "**Key difference**: RoPE encodes position in the Q and K vectors themselves through rotation.")
        elif pos_enc == "alibi":
            st.markdown("**Compute Attention Scores:**")
            st.latex(r"\text{attn\_scores} = \frac{Q K^T}{\sqrt{d_{head}}}")
            st.markdown("**Apply ALiBi (Attention with Linear Biases):**")
            st.markdown(
                "ALiBi adds position-dependent bias AFTER computing attention scores:")
            st.latex(r"m_h = 2^{-8h/n_{heads}}")
            st.latex(r"\text{bias}(i, j) = -m_h \cdot |i - j|")
            st.latex(
                r"\text{attn\_scores} = \text{attn\_scores} + \text{bias\_matrix}")
            st.markdown(
                f"where $h$ is the head index, $n_{{heads}} = {n_heads}$")
            st.markdown(
                "**Key difference**: ALiBi adds position information as a bias term after computing $QK^T$.")
        else:  # learned or none
            st.markdown("**Compute Attention Scores:**")
            st.latex(r"\text{attn\_scores} = \frac{Q K^T}{\sqrt{d_{head}}}")
            st.markdown(
                "**Note**: With learned positional embeddings, position information is already in $x_{\\text{norm}}$ (added at the embedding stage), so attention computation is standard.")

        st.markdown("**Causal Masking and Attention Pattern:**")
        st.latex(
            r"\text{mask}_{i,j} = \begin{cases} 1 & \text{if } i \geq j \\ 0 & \text{if } i < j \end{cases}")
        st.latex(
            r"\text{attn\_scores}_{i,j} = \begin{cases} \text{attn\_scores}_{i,j} & \text{if } i \geq j \\ -\infty & \text{if } i < j \end{cases}")
        st.latex(
            r"\text{attn\_pattern} = \text{softmax}(\text{attn\_scores}, \text{dim}=-1)")
        st.markdown("**Apply to Values and Output Projection:**")
        if n_kv_heads < n_heads:
            st.latex(
                r"\text{attn\_output} = \text{attn\_pattern} \cdot V_{\text{broadcast}}")
        else:
            st.latex(r"\text{attn\_output} = \text{attn\_pattern} \cdot V")
        st.latex(r"\text{attn\_output} = \text{attn\_output} \cdot W_O^T")
        st.markdown("**Residual Connection:**")
        st.latex(r"x = x + \text{attn\_output}")

        # Add note about KV cache for GQA/MQA
        if n_kv_heads < n_heads:
            st.markdown("---")
            st.markdown("**KV Cache Note:**")
            st.markdown(
                f"For efficient inference, the KV cache stores the original (non-broadcasted) K/V: "
                f"$[B, L, n_{{kv\\_heads}}, d_{{head}}]$ = $[B, L, {n_kv_heads}, {d_head}]$ "
                f"instead of $[B, L, n_{{heads}}, d_{{head}}]$ = $[B, L, {n_heads}, {d_head}]$. "
                f"This reduces cache size by {(1 - n_kv_heads/n_heads)*100:.1f}% compared to MHA.")

        st.markdown("#### 3.2 MLP Sub-block")

        if use_moe:
            st.markdown("**Mixture of Experts (MoE) MLP:**")
            st.markdown(
                f"Using {num_experts} expert MLPs, activating top-{num_experts_per_tok} per token.")

            st.markdown("**Router Network:**")
            st.latex(
                r"\text{router\_logits} = x_{\text{norm}} W_{\text{router}}^T")
            st.latex(
                r"\text{router\_probs} = \text{softmax}(\text{router\_logits}, \text{dim}=-1)")
            st.markdown(
                f"where $W_{{\\text{{router}}}} \\in \\mathbb{{R}}^{{{d_model} \\times {num_experts}}}$")

            st.markdown("**Top-k Expert Selection:**")
            st.latex(
                r"\text{top\_k\_probs}, \text{top\_k\_indices} = \text{topk}(\text{router\_probs}, k=" + str(num_experts_per_tok) + r")")
            st.latex(
                r"\text{top\_k\_probs} = \frac{\text{top\_k\_probs}}{\sum_{k} \text{top\_k\_probs}}")

            st.markdown("**Expert Processing:**")
            st.latex(r"\text{output} = \sum_{i=1}^{" + str(num_experts) +
                     r"} w_i \cdot \text{Expert}_i(x_{\text{norm}})")
            st.markdown(
                "where $w_i$ is the routing weight for expert $i$ (0 if not selected, normalized top-k probability if selected)")

            if use_shared_experts:
                st.markdown(
                    f"**Shared Experts ({num_shared_experts} always active):**")
                st.latex(r"\text{shared\_output} = \frac{1}{" + str(num_shared_experts) +
                         r"} \sum_{j=1}^{" + str(num_shared_experts) + r"} \text{SharedExpert}_j(x_{\text{norm}})")
                st.latex(
                    r"\text{output} = \text{output} + \text{shared\_output}")

            st.markdown("**Load Balancing Loss (Auxiliary):**")
            st.latex(
                r"f_i = \frac{\text{tokens routed to expert } i}{\text{total tokens} \times k}")
            st.latex(r"P_i = \text{mean}(\text{router\_probs}[:, :, i])")
            st.latex(r"\mathcal{L}_{\text{aux}} = " + str(num_experts) +
                     r" \times \sum_{i=1}^{" + str(num_experts) + r"} P_i \cdot f_i")
            st.markdown("**Total Loss:**")
            st.latex(r"\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{main}} + " + str(
                load_balancing_loss_weight) + r" \times \mathcal{L}_{\text{aux}}")

            st.markdown("**Expert MLP Architecture:**")
            if activation == "swiglu":
                st.markdown(
                    "Each expert uses SwiGLU activation (same as standard SwiGLU MLP)")
            else:
                st.markdown(
                    "Each expert uses GELU activation (same as standard GELU MLP)")

        elif norm == "layernorm":
            st.markdown("""
            **Pre-Normalization (LayerNorm):**
            ```
            x_norm = LayerNorm(x)  # Same as above
            ```
            """)
        elif norm == "rmsnorm":
            st.markdown("""
            **Pre-Normalization (RMSNorm):**
            ```
            x_norm = RMSNorm(x)  # Same as above
            ```
            """)

        if activation == "gelu":
            st.markdown("**MLP with GELU Activation:**")
            st.latex(
                r"\text{hidden} = x_{\text{norm}} W_{\text{in}}^T + b_{\text{in}}")
            st.markdown("GELU activation function:")
            st.latex(r"\text{GELU}(x) = x \cdot \Phi(x)")
            st.markdown(
                "where $\\Phi(x)$ is the CDF of the standard normal distribution. Approximation:")
            st.latex(
                r"\text{GELU}(x) \approx 0.5x \cdot \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} \cdot (x + 0.044715x^3)\right)\right)")
            st.latex(r"\text{hidden} = \text{GELU}(\text{hidden})")
            st.latex(
                r"\text{mlp\_output} = \text{hidden} \cdot W_{\text{out}}^T + b_{\text{out}}")
            st.latex(r"x = x + \text{mlp\_output}")
            w_in_dim = f"{d_model} \\times {d_mlp}"
            w_out_dim = f"{d_mlp} \\times {d_model}"
            st.markdown(
                f"where $W_{{\\text{{in}}}} \\in \\mathbb{{R}}^{{{w_in_dim}}}$, $W_{{\\text{{out}}}} \\in \\mathbb{{R}}^{{{w_out_dim}}}$")
        elif activation == "swiglu":
            st.markdown("**MLP with SwiGLU Activation:**")
            st.latex(
                r"\text{gate} = x_{\text{norm}} W_{\text{gate}}^T + b_{\text{gate}}")
            st.latex(
                r"\text{up} = x_{\text{norm}} W_{\text{up}}^T + b_{\text{up}}")
            st.markdown("SwiGLU activation (SiLU on gate, multiplied by up):")
            st.latex(
                r"\text{SiLU}(x) = x \cdot \sigma(x) = x \cdot \frac{1}{1 + e^{-x}}")
            st.latex(
                r"\text{hidden} = \text{SiLU}(\text{gate}) \odot \text{up}")
            st.latex(
                r"\text{mlp\_output} = \text{hidden} \cdot W_{\text{out}}^T + b_{\text{out}}")
            st.latex(r"x = x + \text{mlp\_output}")
            w_gate_dim = f"{d_model} \\times {d_mlp}"
            w_out_dim = f"{d_mlp} \\times {d_model}"
            st.markdown(
                f"where $W_{{\\text{{gate}}}}, W_{{\\text{{up}}}} \\in \\mathbb{{R}}^{{{w_gate_dim}}}$, $W_{{\\text{{out}}}} \\in \\mathbb{{R}}^{{{w_out_dim}}}$")

        st.markdown("---")
        st.markdown("### 4. Output Projection")
        if norm == "layernorm":
            st.latex(r"x_{\text{final}} = \text{LayerNorm}(x)")
        else:
            st.latex(r"x_{\text{final}} = \text{RMSNorm}(x)")
        st.latex(r"\text{logits} = x_{\text{final}} W_{\text{unembed}}^T")
        st.latex(r"p = \text{softmax}(\text{logits}, \text{dim}=-1)")
        st.markdown(
            f"where $W_{{\\text{{unembed}}}} \\in \\mathbb{{R}}^{{V \\times {d_model}}}$, $V$ = vocabulary size")

        st.markdown("---")
        st.markdown("### 5. Training Loss (Next-Token Prediction)")
        st.markdown("For each position $i$, predict token at position $i+1$:")
        st.latex(r"\text{Input: } [t_0, t_1, \ldots, t_{n-1}]")
        st.latex(r"\text{Target: } [t_1, t_2, \ldots, t_n]")
        st.markdown("Cross-entropy loss:")
        st.latex(r"\mathcal{L} = -\log p_{i+1}(t_{i+1} | t_0, \ldots, t_i)")
        st.markdown("Average over sequence and batch:")
        st.latex(
            r"\mathcal{L} = -\frac{1}{B \cdot L} \sum_{b=1}^{B} \sum_{i=1}^{L} \log p_{i+1}^{(b)}(t_{i+1}^{(b)})")

        st.markdown("---")
        st.markdown("### Summary")
        moe_info = ""
        if use_moe:
            moe_info = f"\n        - **MoE**: {num_experts} experts, top-{num_experts_per_tok}"
            if use_shared_experts:
                moe_info += f", {num_shared_experts} shared experts"

        st.markdown(f"""
        **Your Model Configuration:**
        - **Positional Encoding**: {pos_enc.upper()}
        - **Normalization**: {norm.upper()}
        - **Activation**: {activation.upper()}{moe_info}
        - **Dimensions**: d_model={d_model}, n_heads={n_heads}, d_head={d_head}, d_mlp={d_mlp}
        """)


def _get_file_relative_path(absolute_path: str) -> str:
    """Convert absolute path to relative path from project root."""
    try:
        # Try to find project root by looking for common markers
        current = Path(absolute_path).resolve()
        project_root = None

        # Walk up the directory tree to find project root
        for _ in range(10):  # Limit search depth
            if (current / "main.py").exists() or (current / "pyproject.toml").exists():
                project_root = current
                break
            if current == current.parent:
                break
            current = current.parent

        if project_root:
            try:
                rel_path = Path(absolute_path).relative_to(project_root)
                return str(rel_path).replace('\\', '/')
            except ValueError:
                pass

        # Fallback: find 'pretraining' or 'finetuning' in path
        parts = Path(absolute_path).parts
        for i, part in enumerate(parts):
            if part in ['pretraining', 'finetuning', 'inference']:
                return '/'.join(parts[i:])

        # Final fallback: return just the filename
        return Path(absolute_path).name
    except Exception:
        return Path(absolute_path).name


def _get_class_source_with_lines(module_path: str, class_name: str, method_name: str = "forward") -> Tuple[str, int, int, str]:
    """
    Extract source code for a class method with line numbers.

    Args:
        module_path: Path to module (e.g., "pretraining.model.model")
        class_name: Name of class (e.g., "TransformerModel")
        method_name: Name of method (default: "forward")

    Returns:
        Tuple of (source_code, start_line, end_line, file_path)
    """
    try:
        # Import module dynamically
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        method = getattr(cls, method_name)

        # Get source lines
        source_lines, start_line = inspect.getsourcelines(method)
        source_code = ''.join(source_lines)
        end_line = start_line + len(source_lines) - 1

        # Get file path
        file_path = inspect.getfile(cls)

        return source_code, start_line, end_line, file_path
    except Exception as e:
        raise Exception(
            f"Could not extract source for {module_path}.{class_name}.{method_name}: {e}")


def _get_object_source_with_lines(module_path: str, object_name: str) -> Tuple[str, int, int, str]:
    """
    Extract source code for a class or function with line numbers.
    Works for both classes and functions since inspect.getsourcelines() handles both.

    Args:
        module_path: Path to module (e.g., "pretraining.model.model")
        object_name: Name of class or function (e.g., "TransformerModel" or "convert_model_to_lora")

    Returns:
        Tuple of (source_code, start_line, end_line, file_path)
    """
    try:
        # Import module dynamically
        module = __import__(module_path, fromlist=[object_name])
        obj = getattr(module, object_name)

        # Get source lines (works for both classes and functions)
        source_lines, start_line = inspect.getsourcelines(obj)
        source_code = ''.join(source_lines)
        end_line = start_line + len(source_lines) - 1

        # Get file path
        file_path = inspect.getfile(obj)

        return source_code, start_line, end_line, file_path
    except Exception as e:
        raise Exception(
            f"Could not extract source for {module_path}.{object_name}: {e}")


def _generate_github_link(
    file_path: str,
    start_line: int,
    end_line: int,
    github_repo_url: str = "https://github.com/jammastergirish/BuildAnLLM",
    branch: str = "main"
) -> str:
    """
    Generate GitHub link to code snippet.

    Format: https://github.com/user/repo/blob/branch/path/to/file.py#L10-L20
    """
    # Convert absolute path to relative
    rel_path = _get_file_relative_path(file_path)

    # Construct URL
    repo_url = github_repo_url.rstrip('/')
    # Replace backslashes with forward slashes for URL
    rel_path_url = rel_path.replace('\\', '/')
    return f"{repo_url}/blob/{branch}/{rel_path_url}#L{start_line}-L{end_line}"


def _determine_components_to_show(config: Dict) -> Dict[str, Dict]:
    """
    Determine which components and classes to show based on config.

    Returns:
        Dict mapping component names to their configuration
    """
    use_einops = config.get("use_einops", True)
    pos_enc = config.get("positional_encoding", "learned")
    norm = config.get("normalization", "layernorm")
    activation = config.get("activation", "gelu")

    components = {}

    # Model (unified class - use_einops is handled internally)
    components["model"] = {
        "class": "TransformerModel",
        "use_einops": use_einops
    }

    # Transformer Block (unified class - use_einops is handled internally)
    components["transformer_block"] = {
        "class": "TransformerBlock",
        "use_einops": use_einops
    }

    # Attention (unified class - use_einops is handled internally)
    components["attention"] = {
        "class": "Attention",
        "pos_enc": pos_enc
    }

    # MLP (unified classes - use_einops is handled internally)
    use_moe = config.get("use_moe", False)
    if use_moe:
        # MoE MLP classes
        components["mlp"] = {
            "class": "MoEMLP",
            "activation": activation,
            "use_moe": True
        }
    elif activation == "swiglu":
        components["mlp"] = {
            "class": "MLPSwiGLU",
            "activation": activation
        }
    else:  # gelu
        components["mlp"] = {
            "class": "MLP",
            "activation": activation
        }

    # Normalization (unified classes - use_einops is handled internally)
    if norm == "rmsnorm":
        components["normalization"] = {
            "class": "RMSNorm",
            "file": "rmsnorm",
            "norm": norm
        }
    else:  # layernorm
        components["normalization"] = {
            "class": "LayerNorm",
            "file": "layernorm",
            "norm": norm
        }

    # Positional Encoding (unified class - use_einops is handled internally)
    if pos_enc == "learned":
        components["positional_encoding"] = {
            "class": "PosEmbed",
            "type": "learned",
            "module": "pretraining.positional_embeddings.positional_embedding"
        }
    elif pos_enc == "rope":
        components["positional_encoding"] = {
            "class": "RoPE",
            "type": "rope",
            "module": "pretraining.positional_embeddings.rope",
            "method": "forward"
        }
    elif pos_enc == "alibi":
        components["positional_encoding"] = {
            "class": "ALiBi",
            "type": "alibi",
            "module": "pretraining.positional_embeddings.alibi",
            "method": "get_bias"  # Show get_bias method for ALiBi
        }
    # else: "none" - don't add positional_encoding

    # Embeddings
    components["embeddings"] = {
        "class": "EmbedWithoutTorch" if use_einops else "EmbedWithTorch",
        "use_einops": use_einops
    }

    return components


def _render_code_section(
    title: str,
    module_path: str,
    class_name: str,
    method_name: str,
    github_repo_url: str = "https://github.com/jammastergirish/BuildAnLLM"
) -> bool:
    """
    Render a single code section with file info and GitHub link.

    Returns:
        True if successful, False otherwise
    """
    try:
        source_code, start_line, end_line, file_path = _get_class_source_with_lines(
            module_path, class_name, method_name
        )
        rel_path = _get_file_relative_path(file_path)

        # Display title and file info
        st.markdown(f"### {title}")
        st.caption(f"📄 `{rel_path}` (lines {start_line}-{end_line})")

        # GitHub link
        github_link = _generate_github_link(
            file_path, start_line, end_line, github_repo_url
        )
        st.markdown(f"🔗 [View on GitHub]({github_link})")

        # Code block
        st.code(source_code, language="python")

        return True
    except Exception as e:
        st.warning(f"Could not load code for {title}: {e}")
        return False


def _render_entire_class(
    title: str,
    module_path: str,
    class_name: str,
    github_repo_url: str = "https://github.com/jammastergirish/BuildAnLLM"
) -> bool:
    """
    Render an entire class definition with all its methods.

    Args:
        title: Section title
        module_path: Path to module
        class_name: Name of class
        github_repo_url: GitHub repo URL

    Returns:
        True if successful, False otherwise
    """
    try:
        source_code, start_line, end_line, file_path = _get_object_source_with_lines(
            module_path, class_name
        )
        rel_path = _get_file_relative_path(file_path)

        # Display title and file info
        st.markdown(f"### {title}")
        st.caption(f"📄 `{rel_path}` (lines {start_line}-{end_line})")

        # GitHub link
        github_link = _generate_github_link(
            file_path, start_line, end_line, github_repo_url
        )
        st.markdown(f"🔗 [View on GitHub]({github_link})")

        # Code block
        st.code(source_code, language="python")

        return True
    except Exception as e:
        st.warning(f"Could not load code for {title}: {e}")
        return False


def render_model_code_snippets(config: Dict) -> None:
    """
    Render relevant code snippets based on model configuration.

    Args:
        config: Model configuration dict
    """
    github_repo_url = "https://github.com/jammastergirish/BuildAnLLM"

    with st.expander("💻 Code", expanded=False):
        components = _determine_components_to_show(config)

        # 1. Model (entire class)
        _render_entire_class(
            "1. Model",
            "pretraining.model.model",
            components["model"]["class"],
            github_repo_url
        )

        st.markdown("---")

        # 2. Transformer Block (entire class)
        _render_entire_class(
            "2. Transformer Block",
            "pretraining.transformer_blocks.transformer_block",
            components["transformer_block"]["class"],
            github_repo_url
        )

        st.markdown("---")

        # 3. Attention Mechanism (entire class)
        _render_entire_class(
            "3. Attention Mechanism",
            "pretraining.attention.attention",
            components["attention"]["class"],
            github_repo_url
        )

        # 4. Positional Encoding (if applicable)
        if "positional_encoding" in components:
            st.markdown("---")
            pos_enc_info = components["positional_encoding"]

            if pos_enc_info["type"] == "learned":
                _render_entire_class(
                    "4. Positional Embeddings (Learned)",
                    pos_enc_info["module"],
                    pos_enc_info["class"],
                    github_repo_url
                )
            elif pos_enc_info["type"] == "rope":
                _render_entire_class(
                    "4. RoPE (Rotary Position Embedding)",
                    pos_enc_info["module"],
                    pos_enc_info["class"],
                    github_repo_url
                )
            elif pos_enc_info["type"] == "alibi":
                _render_entire_class(
                    "4. ALiBi (Attention with Linear Biases)",
                    pos_enc_info["module"],
                    pos_enc_info["class"],
                    github_repo_url
                )

        st.markdown("---")

        # 5. MLP (entire class)
        _render_entire_class(
            "5. MLP",
            "pretraining.mlp.mlp",
            components["mlp"]["class"],
            github_repo_url
        )

        st.markdown("---")

        # 6. Normalization (entire class)
        norm_file = components["normalization"]["file"]
        _render_entire_class(
            "6. Normalization",
            f"pretraining.normalization.{norm_file}",
            components["normalization"]["class"],
            github_repo_url
        )

        st.markdown("---")

        # 7. Embeddings (entire class)
        _render_entire_class(
            "7. Token Embeddings",
            "pretraining.embeddings.embed",
            components["embeddings"]["class"],
            github_repo_url
        )


def _render_function(
    title: str,
    module_path: str,
    function_name: str,
    github_repo_url: str = "https://github.com/jammastergirish/BuildAnLLM"
) -> bool:
    """
    Render a function with file info and GitHub link.
    Uses the same underlying function as _render_entire_class since inspect works for both.

    Args:
        title: Section title
        module_path: Path to module
        function_name: Name of function
        github_repo_url: GitHub repo URL

    Returns:
        True if successful, False otherwise
    """
    try:
        source_code, start_line, end_line, file_path = _get_object_source_with_lines(
            module_path, function_name
        )
        rel_path = _get_file_relative_path(file_path)

        # Display title and file info
        st.markdown(f"### {title}")
        st.caption(f"📄 `{rel_path}` (lines {start_line}-{end_line})")

        # GitHub link
        github_link = _generate_github_link(
            file_path, start_line, end_line, github_repo_url
        )
        st.markdown(f"🔗 [View on GitHub]({github_link})")

        # Code block
        st.code(source_code, language="python")

        return True
    except Exception as e:
        st.warning(f"Could not load code for {title}: {e}")
        return False


def render_inference_equations(
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    top_p: float = 0.9
) -> None:
    """Render mathematical equations for text generation/inference."""
    with st.expander("📐 Equations", expanded=False):
        st.markdown("### Key Notation")
        st.markdown("""
        - **prompt**: Starting text input
        - **logits**: Raw model outputs (before softmax)
        - **probs**: Probability distribution over vocabulary
        - **T**: Temperature (scaling factor)
        - **k**: Top-k sampling parameter
        - **p**: Top-p (nucleus) sampling threshold
        - **V**: Vocabulary size
        """)

        st.markdown("---")
        st.markdown("### 1. Autoregressive Generation")
        st.markdown(
            "Text is generated one token at a time, using previously generated tokens as context:")
        st.latex(r"t_0, t_1, \ldots, t_{n-1} \quad \text{(prompt tokens)}")
        st.latex(
            r"t_n, t_{n+1}, \ldots, t_{n+m-1} \quad \text{(generated tokens)}")
        st.markdown("At each step $i$, the model predicts the next token:")
        st.latex(
            r"\text{logits}_i = \text{model}(t_0, t_1, \ldots, t_{i-1}) \quad \text{logits}_i \in \mathbb{R}^V")
        st.latex(
            r"t_i \sim \text{sample}(\text{logits}_i) \quad \text{(sample next token)}")

        st.markdown("---")
        st.markdown("### 2. Temperature Scaling")
        st.markdown("Temperature controls the randomness of sampling:")
        st.latex(r"\text{logits}_{\text{scaled}} = \frac{\text{logits}}{T}")
        st.latex(
            r"\text{probs} = \text{softmax}(\text{logits}_{\text{scaled}})")
        st.markdown("**Effect:**")
        st.markdown("""
        - **$T < 1$**: Sharper distribution (more focused, less random)
        - **$T = 1$**: Original distribution (no scaling)
        - **$T > 1$**: Flatter distribution (more random, more diverse)
        """)
        st.markdown(f"**Your setting:** $T = {temperature}$")

        if top_k is not None:
            st.markdown("---")
            st.markdown("### 3. Top-k Sampling")
            st.markdown("Only sample from the $k$ most likely tokens:")
            st.latex(
                r"\text{top\_k\_indices} = \text{argsort}(\text{logits})[:k]")
            st.latex(
                r"\text{logits}_i = \begin{cases} \text{logits}_i & \text{if } i \in \text{top\_k\_indices} \\ -\infty & \text{otherwise} \end{cases}")
            st.markdown(
                "**Effect:** Prevents sampling from low-probability tokens, reducing incoherent outputs.")
            st.markdown(f"**Your setting:** $k = {top_k}$")
        else:
            st.markdown("---")
            st.markdown("### 3. Top-k Sampling")
            st.markdown("**Not enabled** (sampling from all tokens)")

        st.markdown("---")
        st.markdown("### 4. Top-p (Nucleus) Sampling")
        st.markdown(
            "Sample from the smallest set of tokens whose cumulative probability exceeds $p$:")
        st.latex(
            r"\text{sorted\_probs} = \text{sort}(\text{probs}, \text{descending=True})")
        st.latex(
            r"\text{cumulative\_probs} = \text{cumsum}(\text{sorted\_probs})")
        st.latex(
            r"\text{min\_set\_size} = \min\{n : \sum_{i=1}^n \text{sorted\_probs}_i \geq p\}")
        st.latex(
            r"\text{logits}_i = \begin{cases} \text{logits}_i & \text{if } i \text{ in min set} \\ -\infty & \text{otherwise} \end{cases}")
        st.markdown(
            "**Effect:** Dynamically adjusts the number of tokens based on distribution shape.")
        st.markdown(f"**Your setting:** $p = {top_p}$")

        st.markdown("---")
        st.markdown("### 5. Final Sampling")
        st.markdown(
            "After applying temperature, top-k, and top-p, sample from the resulting distribution:")
        st.latex(
            r"\text{probs} = \text{softmax}(\text{logits}_{\text{filtered}} / T)")
        st.latex(r"t_{\text{next}} \sim \text{Multinomial}(\text{probs})")
        st.markdown(
            "**In words:** Sample one token according to the filtered probability distribution.")

        st.markdown("---")
        st.markdown("### 6. Complete Generation Process")
        st.markdown("""
        **Algorithm:**
        1. Encode prompt: $\\text{tokens} = \\text{tokenizer.encode}(\\text{prompt})$
        2. For $i = 0$ to $\\text{max\\_new\\_tokens} - 1$:
           - Get logits: $\\text{logits} = \\text{model}(\\text{tokens})[-1]$
           - Apply temperature: $\\text{logits} = \\text{logits} / T$
           - Apply top-k (if enabled): Filter to top $k$ tokens
           - Apply top-p (if enabled): Filter to nucleus set
           - Sample: $t_i \\sim \\text{softmax}(\\text{logits})$
           - Append: $\\text{tokens} = \\text{tokens} + [t_i]$
        3. Decode: $\\text{text} = \\text{tokenizer.decode}(\\text{tokens})$
        """)

        st.markdown("---")
        st.markdown("### Summary")
        st.markdown(f"""
        **Your Configuration:**
        - **Temperature**: $T = {temperature}$ ({'More focused' if temperature < 1 else 'More random' if temperature > 1 else 'Balanced'})
        - **Top-k**: {'Enabled' if top_k is not None else 'Disabled'} {f'($k = {top_k}$)' if top_k is not None else ''}
        - **Top-p**: {'Enabled' if top_p > 0 else 'Disabled'} {f'($p = {top_p}$)' if top_p > 0 else ''}
        """)


def render_inference_code_snippets() -> None:
    """
    Render relevant code snippets for inference/text generation.
    """
    github_repo_url = "https://github.com/jammastergirish/BuildAnLLM"

    with st.expander("💻 Code", expanded=False):
        # Transformer Sampler (entire class)
        _render_entire_class(
            "1. Transformer Sampler",
            "inference.sampler",
            "TransformerSampler",
            github_repo_url
        )


def render_finetuning_code_snippets(use_lora: bool = False) -> None:
    """
    Render relevant code snippets for fine-tuning based on configuration.

    Args:
        use_lora: Whether LoRA is being used
    """
    github_repo_url = "https://github.com/jammastergirish/BuildAnLLM"

    with st.expander("💻 Code", expanded=False):
        # 1. SFT Dataset (entire class)
        _render_entire_class(
            "1. SFT Dataset",
            "finetuning.data.sft_dataset",
            "SFTDataset",
            github_repo_url
        )

        st.markdown("---")

        # 2. SFT Trainer (entire class)
        _render_entire_class(
            "2. SFT Trainer",
            "finetuning.training.sft_trainer",
            "SFTTrainer",
            github_repo_url
        )

        # 3. LoRA (if applicable)
        if use_lora:
            st.markdown("---")
            _render_function(
                "3. LoRA Conversion",
                "finetuning.peft.lora_utils",
                "convert_model_to_lora",
                github_repo_url
            )

            st.markdown("---")
            _render_function(
                "4. LoRA Matrix Creation",
                "finetuning.peft.lora_wrappers",
                "create_lora_matrices",
                github_repo_url
            )

            st.markdown("---")
            _render_function(
                "5. LoRA Einsum Computation",
                "finetuning.peft.lora_wrappers",
                "einsum_with_lora",
                github_repo_url
            )


def render_finetuning_equations(use_lora: bool = False, lora_rank: int = 8, lora_alpha: float = 8.0) -> None:
    """Render mathematical equations for supervised fine-tuning."""
    with st.expander("📐 Equations", expanded=False):
        st.markdown("### Key Notation")
        st.markdown("""
        - **prompt**: Input text/question/instruction
        - **response**: Desired output/answer
        - **m**: Loss mask (1 for response tokens, 0 for prompt tokens)
        - **W**: Base weight matrix (frozen if using LoRA)
        - **A, B**: LoRA adapter matrices (trainable)
        - **r**: LoRA rank
        - **α**: LoRA alpha (scaling factor)
        """)

        st.markdown("---")
        st.markdown("### 1. Sequence Construction")
        st.markdown("Each training example combines prompt and response:")
        st.latex(
            r"\text{sequence} = [\text{prompt\_tokens}] + [\text{response\_tokens}]")
        st.markdown(
            "After tokenization and shifting (for next-token prediction):")
        st.latex(r"X = [t_0, t_1, \ldots, t_{L-2}] \quad \text{(input)}")
        st.latex(
            r"Y = [t_1, t_2, \ldots, t_{L-1}] \quad \text{(target, shifted by 1)}")
        st.markdown("where $L$ is the sequence length (prompt + response).")

        st.markdown("---")
        st.markdown("### 2. Loss Masking (Key Difference from Pre-Training)")
        st.markdown("**Loss Mask Definition:**")
        st.latex(
            r"m_i = \begin{cases} 1 & \text{if token } i \text{ is in response} \\ 0 & \text{if token } i \text{ is in prompt} \end{cases}")
        st.markdown("**Why Mask?**")
        st.markdown("""
        - Prevents model from learning to repeat the prompt
        - Focuses learning on generating good responses
        - Teaches the model to generate, not copy
        """)

        st.markdown("**Masked Loss Computation:**")
        st.latex(
            r"\text{logits} = \text{model}(X) \quad \text{logits} \in \mathbb{R}^{B \times L \times V}")
        st.latex(
            r"\mathcal{L}_{\text{unmasked}} = -\log p_i(t_i | t_0, \ldots, t_{i-1}) \quad \text{(loss per token)}")
        st.latex(
            r"\mathcal{L} = \frac{\sum_{i=1}^{L} m_i \cdot \mathcal{L}_{\text{unmasked}, i}}{\sum_{i=1}^{L} m_i}")
        st.markdown(
            "**In words:** Average loss only over response tokens (where $m_i = 1$).")

        st.markdown("---")
        st.markdown("### 3. Training Objective")

        if use_lora:
            st.markdown("**LoRA Fine-Tuning:**")
            st.markdown("Only LoRA adapter matrices $A$ and $B$ are trained:")
            st.latex(r"A^*, B^* = \arg\min_{A, B} \mathcal{L}(A, B)")
            st.markdown("where base weights $W$ are frozen. Updates:")
            st.latex(r"A_{t+1} = A_t - \eta \nabla_A \mathcal{L}(A_t, B_t)")
            st.latex(r"B_{t+1} = B_t - \eta \nabla_B \mathcal{L}(A_t, B_t)")
            st.markdown(
                "where $\\eta$ (learning rate) is typically **10-100x lower** than pre-training (e.g., $10^{-5}$ vs $10^{-3}$).")
            st.markdown(
                "**Note:** Base weights $W$ remain frozen and are not updated.")
        else:
            st.markdown("**Full Parameter Fine-Tuning:**")
            st.latex(r"\theta^* = \arg\min_{\theta} \mathcal{L}(\theta)")
            st.markdown(
                "where $\\theta$ are all model parameters, updated with:")
            st.latex(
                r"\theta_{t+1} = \theta_t - \eta \nabla_{\theta} \mathcal{L}(\theta_t)")
            st.markdown(
                "where $\\eta$ (learning rate) is typically **10-100x lower** than pre-training (e.g., $10^{-5}$ vs $10^{-3}$).")

        if use_lora:
            st.markdown("---")
            st.markdown("### 4. LoRA (Low-Rank Adaptation)")
            st.markdown(
                "**LoRA modifies weight matrices with low-rank adapters:**")
            st.latex(
                r"W_{\text{effective}} = W + \frac{\alpha}{r} \cdot (B \cdot A)")
            st.markdown("where:")
            st.latex(
                r"W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}} \quad \text{(frozen base weights)}")
            st.latex(
                r"A \in \mathbb{R}^{r \times d_{\text{in}}} \quad \text{(trainable, initialized with Kaiming)}")
            st.latex(
                r"B \in \mathbb{R}^{d_{\text{out}} \times r} \quad \text{(trainable, initialized to zero)}")
            st.markdown(
                f"where $r = {lora_rank}$ (rank), $\\alpha = {lora_alpha}$ (scaling factor).")

            st.markdown("**Forward Pass with LoRA:**")
            st.latex(
                r"\text{output} = x \cdot W^T + \frac{\alpha}{r} \cdot (x \cdot A^T \cdot B^T)")
            st.markdown("**Benefits:**")
            st.markdown("""
            - **Parameter efficiency**: Only $2 \\times r \\times d$ parameters per weight matrix (vs $d_{\\text{out}} \\times d_{\\text{in}}$)
            - **Memory efficient**: Base weights $W$ are frozen
            - **Fast training**: Fewer parameters to update
            """)

        st.markdown("---")
        st.markdown("### 5. Comparison: Pre-Training vs Fine-Tuning")
        st.markdown("""
        | Aspect | Pre-Training | Fine-Tuning |
        |--------|-------------|-------------|
        | **Data** | Raw text | Prompt/response pairs |
        | **Loss** | All tokens | Only response tokens (masked) |
        | **Learning Rate** | Higher (e.g., $10^{-3}$) | Lower (e.g., $10^{-5}$) |
        | **Epochs** | Many (10+) | Few (1-5) |
        | **Objective** | Learn language patterns | Learn instruction following |
        """)

        st.markdown("---")
        st.markdown("### Summary")
        if use_lora:
            st.markdown(f"""
            **Your Configuration:**
            - **Method**: LoRA (Parameter-Efficient Fine-Tuning)
            - **LoRA Rank**: $r = {lora_rank}$
            - **LoRA Alpha**: $\\alpha = {lora_alpha}$
            - **Scaling**: $\\alpha/r = {lora_alpha/lora_rank:.2f}$
            """)
        else:
            st.markdown("""
            **Your Configuration:**
            - **Method**: Full Parameter Fine-Tuning
            - All model parameters are updated
            """)


def parse_timestamp(timestamp_str: str) -> str:
    """Parse YYYYMMDDHHMMSS format to readable datetime string."""
    try:
        dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return timestamp_str


def organize_checkpoints_by_run(checkpoints: List[Dict]) -> List[Tuple[str, List[Dict]]]:
    """Organize checkpoints by run (timestamp directory)."""
    runs = defaultdict(list)
    for ckpt in checkpoints:
        timestamp = ckpt.get("timestamp", "")
        runs[timestamp].append(ckpt)

    # Sort runs by the latest checkpoint's creation time in each run (latest first)
    def get_latest_time(run_checkpoints):
        # Get the maximum creation time from all checkpoints in this run
        times = [ckpt.get("ctime", 0) for ckpt in run_checkpoints]
        return max(times) if times else 0

    sorted_runs = sorted(
        runs.items(), key=lambda x: get_latest_time(x[1]), reverse=True)
    return sorted_runs


def render_checkpoint_selector(
    header: str = "Select Model Checkpoint",
    filter_finetuned: bool = False,
    help_text: Optional[str] = None,
    no_checkpoints_message: Optional[str] = None,
) -> Optional[Dict]:
    """
    Render checkpoint selection UI and return selected checkpoint.

    Args:
        header: Header text for the checkpoint selection section
        filter_finetuned: If True, filter out fine-tuned checkpoints
        help_text: Help text for the run selectbox
        no_checkpoints_message: Message to show when no checkpoints found

    Returns:
        Selected checkpoint dict or None if no checkpoint selected
    """
    st.header(header)
    checkpoints = scan_checkpoints()

    if not checkpoints:
        msg = no_checkpoints_message or "No checkpoints found. Please train a model first."
        st.warning(msg)
        st.stop()
        return None

    # Filter checkpoints if needed
    if filter_finetuned:
        checkpoints = [
            ckpt for ckpt in checkpoints if "sft" not in ckpt["path"]]
        if not checkpoints:
            msg = (no_checkpoints_message or
                   "No pre-trained checkpoints found. Please pre-train a model first.")
            st.warning(msg)
            st.stop()
            return None

    # Organize by run
    sorted_runs = organize_checkpoints_by_run(checkpoints)

    # Select run first
    run_options = []
    run_display_map = {}
    for timestamp, checkpoints_list in sorted_runs:
        formatted_time = parse_timestamp(timestamp)
        num_checkpoints = len(checkpoints_list)
        plural = "s" if num_checkpoints != 1 else ""
        display_text = f"{formatted_time} ({num_checkpoints} checkpoint{plural})"
        run_options.append(display_text)
        run_display_map[display_text] = timestamp

    if not run_options:
        msg = no_checkpoints_message or "No runs found. Please train a model first."
        st.warning(msg)
        st.stop()
        return None

    help_txt = help_text or "Select a training run to view its checkpoints"
    selected_run_display = st.selectbox(
        "Choose a training run",
        options=run_options,
        help=help_txt
    )

    selected_run_timestamp = run_display_map[selected_run_display]

    # Get checkpoints for selected run
    run_checkpoints = next(
        checkpoints for timestamp, checkpoints in sorted_runs
        if timestamp == selected_run_timestamp
    )

    # Sort checkpoints: final_model.pt first, then by iteration number
    def sort_key(ckpt):
        path = ckpt["path"]
        if "final_model.pt" in path:
            return (0, 0)  # Final model comes first
        # Extract iteration number from checkpoint_XXXX.pt
        try:
            iter_num = int(Path(path).stem.split("_")[1])
            return (1, iter_num)
        except (IndexError, ValueError):
            return (2, 0)  # Unknown format comes last

    run_checkpoints.sort(key=sort_key, reverse=True)

    # Select checkpoint within run
    checkpoint_options = []
    for ckpt in run_checkpoints:
        path = Path(ckpt["path"])
        is_finetuned = ckpt.get("is_finetuned", False)

        if "final_model.pt" in path.name:
            if is_finetuned:
                label = "🏁 Final Model (Fine-tuned)"
            else:
                label = "🏁 Final Model (Pre-trained)"
            checkpoint_options.append((ckpt, label))
        else:
            # Extract iteration number
            try:
                iter_num = int(path.stem.split("_")[1])
                prefix = "Fine-tuned " if is_finetuned else ""
                checkpoint_options.append(
                    (ckpt, f"{prefix}Checkpoint {iter_num:,}"))
            except (IndexError, ValueError):
                prefix = "Fine-tuned " if is_finetuned else ""
                checkpoint_options.append((ckpt, f"{prefix}{path.stem}"))

    selected_checkpoint_idx = st.selectbox(
        "Choose a checkpoint",
        range(len(checkpoint_options)),
        format_func=lambda x: checkpoint_options[x][1],
        help="Select a checkpoint from this training run"
    )

    selected_checkpoint = checkpoint_options[selected_checkpoint_idx][0]

    # Display selected checkpoint info
    checkpoint_name = checkpoint_options[selected_checkpoint_idx][1]
    run_time = selected_run_display.split(" (")[0]
    st.info(
        f"📌 Selected: **{checkpoint_name}** from training run **{run_time}**")

    return selected_checkpoint
