"""Inference page for generating text from trained models."""

import streamlit as st
import torch
import os
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Add parent directory to path to import from main
sys.path.insert(0, str(Path(__file__).parent.parent))

from pretraining.model.model import TransformerModelWithEinops, TransformerModelWithoutEinops
from pretraining.tokenization.tokenizer import CharacterTokenizer, SimpleBPETokenizer, BPETokenizer, SentencePieceTokenizer
from inference.sampler import TransformerSampler


def parse_timestamp(timestamp_str):
    """Parse YYYYMMDDHHMMSS format to readable datetime string."""
    try:
        dt = datetime.strptime(timestamp_str, "%Y%m%d%H%M%S")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        # If parsing fails, return original string
        return timestamp_str


def organize_checkpoints_by_run(checkpoints):
    """Organize checkpoints by run (timestamp directory)."""
    runs = defaultdict(list)
    for ckpt in checkpoints:
        timestamp = ckpt.get("timestamp", "")
        runs[timestamp].append(ckpt)
    
    # Sort runs by timestamp (newest first)
    sorted_runs = sorted(runs.items(), key=lambda x: x[0], reverse=True)
    
    return sorted_runs


st.title("🎯 Inference")

# Checkpoint selection
st.header("1. Select Model Checkpoint")
checkpoints = st.session_state.scan_checkpoints()

if not checkpoints:
    st.warning("No checkpoints found. Please train a model first.")
    st.stop()

# Organize checkpoints by run
runs = organize_checkpoints_by_run(checkpoints)

# Select run first
run_options = []
run_display_map = {}
for timestamp, checkpoints_list in runs:
    formatted_time = parse_timestamp(timestamp)
    num_checkpoints = len(checkpoints_list)
    display_text = f"{formatted_time} ({num_checkpoints} checkpoint{'s' if num_checkpoints != 1 else ''})"
    run_options.append(display_text)
    run_display_map[display_text] = timestamp

if not run_options:
    st.warning("No runs found. Please train a model first.")
    st.stop()

selected_run_display = st.selectbox(
    "Choose a training run",
    options=run_options,
    help="Select a training run to view its checkpoints"
)

selected_run_timestamp = run_display_map[selected_run_display]

# Get checkpoints for selected run
run_checkpoints = next(checkpoints for timestamp, checkpoints in runs if timestamp == selected_run_timestamp)

# Sort checkpoints: final_model.pt first, then by iteration number
def sort_key(ckpt):
    path = ckpt["path"]
    if "final_model.pt" in path:
        return (0, 0)  # Final model comes first
    else:
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
    if "final_model.pt" in path.name:
        checkpoint_options.append((ckpt, "🏁 Final Model"))
    else:
        # Extract iteration number
        try:
            iter_num = int(path.stem.split("_")[1])
            checkpoint_options.append((ckpt, f"Checkpoint {iter_num:,}"))
        except (IndexError, ValueError):
            checkpoint_options.append((ckpt, path.stem))

selected_checkpoint_idx = st.selectbox(
    "Choose a checkpoint",
    range(len(checkpoint_options)),
    format_func=lambda x: checkpoint_options[x][1],
    help="Select a checkpoint from this training run"
)

selected_checkpoint = checkpoint_options[selected_checkpoint_idx][0]

# Display selected checkpoint info
checkpoint_name = checkpoint_options[selected_checkpoint_idx][1]
# Extract just the date/time part from the run display (before the parenthesis)
run_time = selected_run_display.split(" (")[0]
st.info(f"📌 Selected: **{checkpoint_name}** from training run **{run_time}**")

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
            elif tokenizer_type == "bpe-simple":
                # Simple BPE requires original training text to recreate
                if os.path.exists("training.txt"):
                    with open("training.txt", "r", encoding="utf-8") as f:
                        text = f.read()
                    # Use vocab size from config if available
                    vocab_size = cfg.d_vocab if hasattr(cfg, 'd_vocab') else 1000
                    tokenizer = SimpleBPETokenizer(text, vocab_size=vocab_size)
                else:
                    st.error(
                        "training.txt not found. Cannot recreate Simple BPE tokenizer.")
                    st.stop()
            elif tokenizer_type == "bpe-tiktoken" or tokenizer_type == "bpe":
                # Support "bpe" for backward compatibility with old checkpoints
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
                # Helper function to safely get enum value
                def get_enum_value(enum_obj):
                    if enum_obj is None:
                        return "None"
                    if hasattr(enum_obj, 'value'):
                        return enum_obj.value
                    # Handle string values (from deserialized configs)
                    if isinstance(enum_obj, str):
                        return enum_obj
                    return str(enum_obj)
                
                # Get positional encoding value for RoPE check
                pos_enc_value = get_enum_value(cfg.positional_encoding)
                norm_value = get_enum_value(cfg.normalization)
                activation_value = get_enum_value(cfg.activation)
                
                model_details = {
                    "Positional Encoding": pos_enc_value,
                    "Normalization": norm_value,
                    "Activation": activation_value,
                    "d_model": cfg.d_model,
                    "n_layers": cfg.n_layers,
                    "n_heads": cfg.n_heads,
                    "d_head": cfg.d_head,
                    "d_mlp": cfg.d_mlp,
                    "n_ctx": cfg.n_ctx,
                    "d_vocab": cfg.d_vocab,
                    "Parameters": f"{param_count:.2f}M"
                }
                # Add RoPE theta if using RoPE
                if pos_enc_value == "rope":
                    model_details["rope_theta"] = cfg.rope_theta
                # Add tokenizer type from checkpoint
                if "tokenizer_type" in checkpoint:
                    model_details["Tokenizer"] = checkpoint["tokenizer_type"]
                st.json(model_details)

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

