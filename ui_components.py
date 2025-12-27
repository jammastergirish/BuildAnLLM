"""Reusable Streamlit UI components."""

import inspect
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st


# ============================================================================
# Training Time Utilities
# ============================================================================


def format_elapsed_time(seconds: float) -> str:
    """Format elapsed time in a human-readable format.
    
    Args:
        seconds: Elapsed time in seconds
        
    Returns:
        Formatted string (e.g., "45.2s", "5m 30s", "2h 15m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def get_elapsed_time() -> float:
    """Get current elapsed training time from session state.
    
    Returns:
        Elapsed time in seconds, or 0.0 if training hasn't started
    """
    if "training_start_time" in st.session_state:
        return time.time() - st.session_state.training_start_time
    return 0.0


def get_total_training_time() -> float:
    """Get total training time from session state.
    
    Uses training_end_time if available, otherwise calculates from current time.
    
    Returns:
        Total time in seconds, or 0.0 if training hasn't started
    """
    if "training_start_time" not in st.session_state:
        return 0.0
    
    if "training_end_time" in st.session_state:
        return st.session_state.training_end_time - st.session_state.training_start_time
    else:
        return time.time() - st.session_state.training_start_time


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
            eta_str = format_elapsed_time(eta_seconds) if eta_seconds > 0 else "Calculating..."
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
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly"
    )
    st.plotly_chart(fig_all, use_container_width=True)


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
        template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly"
    )
    st.plotly_chart(fig, use_container_width=True)


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
            template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly"
        )
        st.plotly_chart(fig, use_container_width=True)


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
    """Apply model size preset to config."""
    preset = MODEL_SIZE_PRESETS[size]
    for key, value in preset.items():
        config[key] = value


def apply_architecture_preset(preset_name: str, config: Dict) -> None:
    """Apply architecture preset (GPT, LLaMA, OLMo) to config."""
    if preset_name == "GPT":
        config["positional_encoding"] = "learned"
        config["normalization"] = "layernorm"
        config["activation"] = "gelu"
        config["tokenizer_type"] = "bpe-tiktoken"
    elif preset_name == "LLAMA":
        config["positional_encoding"] = "rope"
        config["normalization"] = "rmsnorm"
        config["activation"] = "swiglu"
        config["tokenizer_type"] = "sentencepiece"
        config["rope_theta"] = 10000.0
    elif preset_name == "OLMO":
        config["positional_encoding"] = "alibi"
        config["normalization"] = "layernorm"
        config["activation"] = "swiglu"
        config["tokenizer_type"] = "sentencepiece"


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
        "n_layers": 4,
        "n_ctx": 256,
        "d_head": 64,
        "d_mlp": 1024,
        "rope_theta": 10000.0,
        "tokenizer_type": "bpe-tiktoken",
    }


def _render_preset_buttons(config: Dict) -> None:
    """Render architecture preset buttons."""
    st.subheader("Quick Presets")
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])

    with col1:
        if st.button("🚀 GPT-2", use_container_width=True):
            apply_architecture_preset("GPT", config)
            apply_model_size_preset(config.get("model_size", "small"), config)
            st.rerun()

    with col2:
        if st.button("🦙 LLaMA", use_container_width=True):
            apply_architecture_preset("LLAMA", config)
            apply_model_size_preset(config.get("model_size", "small"), config)
            st.rerun()

    with col3:
        if st.button("🔬 OLMo", use_container_width=True):
            apply_architecture_preset("OLMO", config)
            apply_model_size_preset(config.get("model_size", "small"), config)
            st.rerun()

    with col4:
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
        if st.button("🔹 Small", use_container_width=True, type=button_type,
                     help="Small model: 256 dim, 4 heads, 4 layers"):
            if current_size != "small":
                config["model_size"] = "small"
                apply_model_size_preset("small", config)
                st.rerun()
    
    with col2:
        button_type = "primary" if current_size == "medium" else "secondary"
        if st.button("🔸 Medium", use_container_width=True, type=button_type,
                     help="Medium model: 512 dim, 8 heads, 6 layers"):
            if current_size != "medium":
                config["model_size"] = "medium"
                apply_model_size_preset("medium", config)
                st.rerun()
    
    with col3:
        button_type = "primary" if current_size == "full" else "secondary"
        if st.button("🔶 Full", use_container_width=True, type=button_type,
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


def _get_preset_info() -> str:
    """Get preset information markdown."""
    return """
    **Preset Configurations:**
    - **GPT-2**: Learned positional embeddings, LayerNorm, GELU activation, BPE-tiktoken (GPT-2 style)
    - **LLaMA**: RoPE positional encoding, RMSNorm, SwiGLU activation, SentencePiece tokenizer
    - **OLMo**: ALiBi positional encoding, LayerNorm, SwiGLU activation, SentencePiece tokenizer

    **Model Size:**
    - Controls model dimensions (d_model, n_heads, n_layers, etc.)
    - All presets use the same dimensions for each size
    - Clicking a preset uses the currently selected model size

    **Customization:**
    - All options below can be manually adjusted after selecting a preset
    - Tokenizer is automatically set but can be changed
    """


def generate_graphviz_architecture(config: Dict) -> str:
    """Generate Graphviz DOT code for transformer architecture."""
    n_layers = config.get("n_layers", 4)
    n_heads = config.get("n_heads", 4)
    pos_enc = config.get("positional_encoding", "learned")
    activation = config.get("activation", "gelu")

    # Start building the DOT code
    dot = []
    dot.append('digraph TransformerArchitecture {')
    dot.append('    bgcolor="black";')
    dot.append('    rankdir=BT;')  # Bottom to top like the reference
    dot.append('    nodesep=0.3;')
    dot.append('    ranksep=0.8;')

    # Node styles
    dot.append(
        '    node [shape=box, style=filled, fillcolor="#5a5a5a", fontcolor="white", ')
    dot.append('          fontname="Arial", fontsize=10, height=0.5, width=1.2];')
    dot.append('    edge [color="#aaaaaa", penwidth=1.5, arrowsize=0.7];')
    dot.append('')

    # Create nodes
    dot.append('    // Input/Output nodes')
    dot.append('    tokens [label="tokens", fillcolor="#4a4a4a"];')
    dot.append('    embed [label="embed", fillcolor="#6a6a4a"];')

    # Positional embedding node if needed
    if pos_enc == "learned":
        dot.append(
            '    pos_emb [label="Positional Embeddings", fillcolor="#7a7a4a"];')

    dot.append('    logits [label="logits", fillcolor="#4a4a4a"];')
    dot.append('    unembed [label="unembed", fillcolor="#6a6a4a"];')

    # Create x nodes (residual stream points)
    dot.append('')
    dot.append('    // Residual stream points')
    dot.append('    x0 [shape=plaintext, label="x₀", fontcolor="#cccccc"];')
    dot.append(
        '    x1 [shape=plaintext, label="x_{i+1}", fontcolor="#cccccc"];')
    dot.append(
        '    x2 [shape=plaintext, label="x_{i+2}", fontcolor="#cccccc"];')
    dot.append(
        '    x_final [shape=plaintext, label="x_{-1}", fontcolor="#cccccc"];')

    # Residual block in a cluster
    dot.append('')
    dot.append('    // One residual block (repeated)')
    dot.append('    subgraph cluster_block {')
    dot.append('        style=dashed;')
    dot.append('        color="#ffff88";')
    dot.append('        penwidth=2;')
    dot.append(f'        label="×{n_layers}";')
    dot.append('        fontcolor="#ffff88";')
    dot.append('        fontsize=14;')
    dot.append('        ')

    # Attention heads
    heads_label = f"h₀  h₁  ...  h_{n_heads-1}"
    if pos_enc == "rope":
        heads_label += "\\n(RoPE)"
    elif pos_enc == "alibi":
        heads_label += "\\n(ALiBi)"

    dot.append(f'        heads [label="{heads_label}", fillcolor="#6a5a5a"];')

    # MLP
    mlp_label = "MLP  m"
    if activation == "swiglu":
        mlp_label += "\\n(SwiGLU)"
    elif activation == "gelu":
        mlp_label += "\\n(GELU)"

    dot.append(f'        mlp [label="{mlp_label}", fillcolor="#5a6a5a"];')
    dot.append('    }')

    dot.append('')
    dot.append('    // Connections')

    # Input flow
    dot.append('    tokens -> embed;')

    # Handle positional encoding
    if pos_enc == "learned":
        dot.append('    embed -> pos_emb;')
        dot.append('    pos_emb -> x0;')
    else:
        dot.append('    embed -> x0;')

    # One block connections
    dot.append('    x0 -> x1;')
    dot.append(
        '    x1 -> heads [dir=both, label="+", fontsize=10, fontcolor="yellow"];')
    dot.append('    x1 -> x2;')
    dot.append(
        '    x2 -> mlp [dir=both, label="+", fontsize=10, fontcolor="yellow"];')

    # Repetition indicator
    dot.append(
        '    x2 -> x_final [label="...", fontsize=12, fontcolor="#888888"];')

    # Output
    dot.append('    x_final -> unembed;')
    dot.append('    unembed -> logits;')

    dot.append('}')

    return '\n'.join(dot)


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
        d_head = config.get("d_head", 64)
        d_mlp = config.get("d_mlp", 1024)
        pos_enc = config.get("positional_encoding", "learned")
        norm = config.get("normalization", "layernorm")
        activation = config.get("activation", "gelu")
        rope_theta = config.get("rope_theta", 10000.0)

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

        st.markdown("**Multi-Head Attention:**")
        st.markdown("Project to Q, K, V for all heads:")
        st.latex(
            r"Q = x_{\text{norm}} W_Q^T \quad Q \in \mathbb{R}^{B \times L \times n_{heads} \times d_{head}}")
        st.latex(
            r"K = x_{\text{norm}} W_K^T \quad K \in \mathbb{R}^{B \times L \times n_{heads} \times d_{head}}")
        st.latex(
            r"V = x_{\text{norm}} W_V^T \quad V \in \mathbb{R}^{B \times L \times n_{heads} \times d_{head}}")
        st.markdown(
            f"where $W_Q, W_K, W_V \\in \\mathbb{{R}}^{{{n_heads} \\times {d_head} \\times {d_model}}}$")

        if pos_enc == "rope":
            st.markdown("**Apply RoPE (Rotary Position Embedding):**")
            st.markdown(
                "RoPE rotates Q and K vectors BEFORE computing attention scores:")
            st.latex(
                r"R_i = \begin{bmatrix} \cos(\theta_d \cdot i) & -\sin(\theta_d \cdot i) \\ \sin(\theta_d \cdot i) & \cos(\theta_d \cdot i) \end{bmatrix}")
            st.latex(
                r"Q_{\text{rotated}} = R_i \cdot Q \quad K_{\text{rotated}} = R_j \cdot K")
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
        st.latex(r"\text{attn\_output} = \text{attn\_pattern} \cdot V")
        st.latex(r"\text{attn\_output} = \text{attn\_output} \cdot W_O^T")
        st.markdown("**Residual Connection:**")
        st.latex(r"x = x + \text{attn\_output}")

        st.markdown("#### 3.2 MLP Sub-block")

        if norm == "layernorm":
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
        st.markdown(f"""
        **Your Model Configuration:**
        - **Positional Encoding**: {pos_enc.upper()}
        - **Normalization**: {norm.upper()}
        - **Activation**: {activation.upper()}
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
        class_name: Name of class (e.g., "TransformerModelWithEinops")
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
        object_name: Name of class or function (e.g., "TransformerModelWithEinops" or "convert_model_to_lora")

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

    # Model
    components["model"] = {
        "class": "TransformerModelWithEinops" if use_einops else "TransformerModelWithoutEinops",
        "use_einops": use_einops
    }

    # Transformer Block
    components["transformer_block"] = {
        "class": "TransformerBlockWithEinops" if use_einops else "TransformerBlockWithoutEinops",
        "use_einops": use_einops
    }

    # Attention
    components["attention"] = {
        "class": "AttentionWithEinops" if use_einops else "AttentionWithoutEinops",
        "pos_enc": pos_enc
    }

    # MLP
    if activation == "swiglu":
        components["mlp"] = {
            "class": "MLPSwiGLUWithEinops" if use_einops else "MLPSwiGLUWithoutEinops",
            "activation": activation
        }
    else:  # gelu
        components["mlp"] = {
            "class": "MLPWithEinops" if use_einops else "MLPWithoutEinops",
            "activation": activation
        }

    # Normalization
    if norm == "rmsnorm":
        components["normalization"] = {
            "class": "RMSNormWithEinops" if use_einops else "RMSNormWithoutEinops",
            "file": "rmsnorm",
            "norm": norm
        }
    else:  # layernorm
        components["normalization"] = {
            "class": "LayerNormWithEinops" if use_einops else "LayerNormWithoutEinops",
            "file": "layernorm",
            "norm": norm
        }

    # Positional Encoding
    if pos_enc == "learned":
        components["positional_encoding"] = {
            "class": "PosEmbedWithEinops" if use_einops else "PosEmbedWithoutEinops",
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

    sorted_runs = sorted(runs.items(), key=lambda x: get_latest_time(x[1]), reverse=True)
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
    checkpoints = st.session_state.scan_checkpoints()

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
