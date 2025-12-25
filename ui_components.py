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
        config["tokenizer_type"] = "bpe"
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
        "tokenizer_type": "bpe",
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
        index=["small", "medium", "full"].index(config.get("model_size", "small")),
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
    - **GPT-2**: Learned positional embeddings, LayerNorm, GELU activation, BPE tokenizer
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

