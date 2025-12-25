"""Inference page for generating text from trained models."""

import streamlit as st
import torch
import os
import sys
from pathlib import Path

# Add parent directory to path to import from main
sys.path.insert(0, str(Path(__file__).parent.parent))

from model import TransformerModelWithEinops, TransformerModelWithoutEinops
from tokenizer import CharacterTokenizer, BPETokenizer, SentencePieceTokenizer
from sampler import TransformerSampler


st.title("🎯 Inference")

# Checkpoint selection
st.header("1. Select Model Checkpoint")
checkpoints = st.session_state.scan_checkpoints()

if not checkpoints:
    st.warning("No checkpoints found. Please train a model first.")
    st.stop()

checkpoint_options = [ckpt["name"] for ckpt in checkpoints]
selected_idx = st.selectbox(
    "Choose a checkpoint",
    range(len(checkpoint_options)),
    format_func=lambda x: checkpoint_options[x]
)

selected_checkpoint = checkpoints[selected_idx]
st.info(f"Selected: {selected_checkpoint['name']}")

# Load model button
load_model = st.button("📥 Load Model", type="primary")

if load_model or st.session_state.current_model is not None:
    if load_model:
            with st.spinner("Loading model..."):
                device = st.session_state.get_device()
                model, cfg, checkpoint = st.session_state.load_model_from_checkpoint(
                    selected_checkpoint["path"], device
                )

            tokenizer_type = checkpoint.get("tokenizer_type", "character")

            # Create tokenizer
            if tokenizer_type == "character":
                # Need text file for character tokenizer
                if os.path.exists("training.txt"):
                    with open("training.txt", "r", encoding="utf-8") as f:
                        text = f.read()
                    tokenizer = CharacterTokenizer(text)
                else:
                    st.error(
                        "training.txt not found. Cannot load character tokenizer.")
                    st.stop()
            elif tokenizer_type == "bpe":
                tokenizer = BPETokenizer()
            elif tokenizer_type == "sentencepiece":
                # SentencePiece requires original training text to recreate
                if os.path.exists("training.txt"):
                    with open("training.txt", "r", encoding="utf-8") as f:
                        text = f.read()
                    # Use vocab size from config if available
                    vocab_size = cfg.d_vocab if hasattr(
                        cfg, 'd_vocab') else 10000
                    tokenizer = SentencePieceTokenizer(
                        text, vocab_size=vocab_size)
                else:
                    st.error(
                        "training.txt not found. Cannot recreate SentencePiece tokenizer.")
                    st.stop()
            else:
                st.error(
                    f"Tokenizer type {tokenizer_type} not supported in this UI.")
                st.stop()

            st.session_state.current_model = model
            st.session_state.current_tokenizer = tokenizer
            st.session_state.current_cfg = cfg

            param_count = sum(p.numel() for p in model.parameters()) / 1e6
            st.success(
                f"✅ Model loaded: {param_count:.2f}M parameters, {tokenizer_type} tokenizer")

            # Show model details
            with st.expander("📋 Model Details", expanded=False):
                st.json({
                    "Architecture": cfg.architecture.value if hasattr(cfg.architecture, 'value') else str(cfg.architecture),
                    "d_model": cfg.d_model,
                    "n_layers": cfg.n_layers,
                    "n_heads": cfg.n_heads,
                    "d_head": cfg.d_head,
                    "d_mlp": cfg.d_mlp,
                    "n_ctx": cfg.n_ctx,
                    "d_vocab": cfg.d_vocab,
                    "Parameters": f"{param_count:.2f}M"
                })

    if st.session_state.current_model is not None:
        # Inference controls
        st.header("2. Generation Settings")

        col1, col2 = st.columns(2)

        with col1:
            prompt = st.text_area(
                "Prompt",
                value="First Citizen:",
                height=100,
                help="Enter your starting prompt here"
            )
            max_new_tokens = st.number_input(
                "Max New Tokens",
                min_value=1,
                max_value=1000,
                value=200,
                help="Maximum number of tokens to generate"
            )

        with col2:
            temperature = st.slider(
                "Temperature",
                min_value=0.1,
                max_value=2.0,
                value=0.8,
                step=0.1,
                help="Higher = more random, Lower = more focused"
            )
            top_k = st.number_input(
                "Top-k (optional)",
                min_value=1,
                max_value=100,
                value=None,
                help="Only sample from top k tokens (None to disable)"
            )
            top_p = st.slider(
                "Top-p (Nucleus)",
                min_value=0.0,
                max_value=1.0,
                value=0.9,
                step=0.1,
                help="Cumulative probability threshold (None to disable)"
            )

        # Generate button
        generate = st.button(
            "✨ Generate Text", type="primary", width='stretch')

        if generate:
            with st.spinner("Generating text..."):
                sampler = TransformerSampler(
                    model=st.session_state.current_model,
                    tokenizer=st.session_state.current_tokenizer,
                    device=st.session_state.get_device()
                )

                generated = sampler.sample(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k if top_k else None,
                    top_p=top_p if top_p > 0 else None
                )

            st.header("3. Generated Text")
            st.text_area(
                "Output",
                value=generated,
                height=300,
                label_visibility="collapsed"
            )

            # Show prompt separately
            st.caption(f"Prompt: {prompt}")
            st.caption(
                f"Generated: {len(generated) - len(prompt)} characters")

