"""Reusable Streamlit UI components."""

import streamlit as st
from typing import Dict


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

    # Model components
    _render_model_components(config)

    # Model dimensions
    _render_model_dimensions(config)

    # Model size selector
    _render_model_size_selector(config)

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
    """Render model size selector."""
    st.markdown("**Model Size Preset**")
    model_size = st.selectbox(
        "Size",
        ["small", "medium", "full"],
        index=["small", "medium", "full"].index(
            config.get("model_size", "small")),
        help="Selecting a size automatically updates all model dimensions below."
    )

    if config.get("model_size") != model_size:
        config["model_size"] = model_size
        apply_model_size_preset(model_size, config)
        st.rerun()

    config["model_size"] = model_size


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
