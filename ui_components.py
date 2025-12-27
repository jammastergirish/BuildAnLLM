"""Reusable Streamlit UI components."""

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st


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

    sorted_runs = sorted(runs.items(), key=lambda x: x[0], reverse=True)
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
