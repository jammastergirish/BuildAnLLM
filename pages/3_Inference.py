"""Inference page for generating text from trained models."""

from ui_components import render_checkpoint_selector, render_inference_equations, render_inference_code_snippets, render_attention_heatmap
from pretraining.tokenization.tokenizer import (
    BPETokenizer,
    CharacterTokenizer,
    SentencePieceTokenizer,
    SimpleBPETokenizer,
)
from pretraining.model.model import TransformerModel
from pretraining.model.model_loader import load_model_from_checkpoint
from inference.sampler import TransformerSampler
from utils import get_device
import os
import sys
from pathlib import Path

import streamlit as st
import torch
import plotly.express as px
import pandas as pd
import numpy as np

# Add parent directory to path to import from main
sys.path.insert(0, str(Path(__file__).parent.parent))


st.title("🎯 Inference")

# Initialize session state for current model
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "current_tokenizer" not in st.session_state:
    st.session_state.current_tokenizer = None

# Checkpoint selection
selected_checkpoint = render_checkpoint_selector(
    header="1. Select Model Checkpoint",
    filter_finetuned=False,
    help_text="Select a training run to view its checkpoints (includes both pre-trained and fine-tuned models)",
    no_checkpoints_message="No checkpoints found. Please train a model first."
)

# Load model button
load_model = st.button("📥 Load Model", type="primary")

if load_model or st.session_state.current_model is not None:
    if load_model:
        with st.spinner("Loading model..."):
            device = get_device()
            model, cfg, checkpoint = load_model_from_checkpoint(
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
            # Check if tokenizer model exists in checkpoint directory
            checkpoint_dir = os.path.dirname(selected_checkpoint["path"])
            tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.model")
            
            if os.path.exists(tokenizer_path):
                tokenizer = SimpleBPETokenizer(model_path=tokenizer_path)
            # Simple BPE requires original training text to recreate
            elif os.path.exists("training.txt"):
                with open("training.txt", "r", encoding="utf-8") as f:
                    text = f.read()
                # Use vocab size from config if available
                vocab_size = cfg.d_vocab if hasattr(cfg, 'd_vocab') else 1000
                tokenizer = SimpleBPETokenizer(text, vocab_size=vocab_size)
            else:
                st.error(
                    "training.txt not found (and no tokenizer.model in checkpoint). Cannot recreate Simple BPE tokenizer.")
                st.stop()
        elif tokenizer_type == "bpe-tiktoken" or tokenizer_type == "bpe":
            # Support "bpe" for backward compatibility with old checkpoints
            tokenizer = BPETokenizer()
        elif tokenizer_type == "sentencepiece":
            # Check if tokenizer model exists in checkpoint directory
            checkpoint_dir = os.path.dirname(selected_checkpoint["path"])
            tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.model")
            
            if os.path.exists(tokenizer_path):
                tokenizer = SentencePieceTokenizer(model_path=tokenizer_path)
            # Fallback: SentencePiece requires original training text to recreate
            elif os.path.exists("training.txt"):
                with open("training.txt", "r", encoding="utf-8") as f:
                    text = f.read()
                # Use vocab size from config if available
                vocab_size = cfg.d_vocab if hasattr(
                    cfg, 'd_vocab') else 10000
                tokenizer = SentencePieceTokenizer(
                    text, vocab_size=vocab_size)
            else:
                st.error(
                    "training.txt not found (and no tokenizer.model in checkpoint). Cannot recreate SentencePiece tokenizer.")
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
        st.header("2. Settings")

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

        # Understand generation
        st.header("2. Understand Text Generation")

        # Show code implementation (doesn't depend on settings)
        render_inference_code_snippets()

        # Show equations (after settings so values are available)
        render_inference_equations(
            temperature=temperature,
            top_k=top_k if top_k else None,
            top_p=top_p if top_p > 0 else 0.0
        )

        # Generate button
        generate = st.button(
            "✨ Generate Text", type="primary", width='stretch')

        if generate:
            with st.spinner("Generating text..."):
                sampler = TransformerSampler(
                    model=st.session_state.current_model,
                    tokenizer=st.session_state.current_tokenizer,
                    device=get_device()
                )

                generated = sampler.sample(
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k if top_k else None,
                    top_p=top_p if top_p > 0 else None
                )
                
                # Store in session state
                st.session_state.last_generated_text = generated
                st.session_state.last_prompt = prompt

            # === INTERNALS VISUALIZATION ===
            with st.spinner("Analyzing model internals for the input prompt..."):
                # Run a forward pass on the PROMPT (Input) text to get diagnostics
                # User requested to see attention on input text, not the generated text
                full_tokens = st.session_state.current_tokenizer.encode_tensor(prompt).to(get_device()).unsqueeze(0)
                
                with torch.no_grad():
                    # We only care about the last token prediction essentially, but let's capture the whole sequence
                    # Note: We need to use the model directly, not the sampler, to get diagnostics
                    outputs = st.session_state.current_model(
                        full_tokens, 
                        return_diagnostics=True
                    )
                    # Handle return format (logits, cache, aux, diagnostics)
                    if isinstance(outputs, tuple):
                        diagnostics = outputs[-1]
                    else:
                        diagnostics = None 
                    
                    st.session_state.last_diagnostics = diagnostics
                    st.session_state.last_full_tokens = full_tokens

        # Display results (persisted)
        if "last_generated_text" in st.session_state:
            generated = st.session_state.last_generated_text
            prompt_used = st.session_state.get("last_prompt", "")
            
            st.header("4. Generated Text")
            st.text_area(
                "Output",
                value=generated,
                height=300,
                label_visibility="collapsed"
            )
            st.caption(f"Prompt: {prompt_used}")
            st.caption(
                f"Generated: {len(generated) - len(prompt_used)} characters")

            # === INTERNALS VISUALIZATION ===
            st.header("5. 🧠 Model Internals (Glass Box)")
            
            diagnostics = st.session_state.get("last_diagnostics")
            full_tokens = st.session_state.get("last_full_tokens")
            
            if diagnostics:
                internals_tabs = st.tabs(["🔥 Attention Heatmaps", "🔍 Logit Lens", "📊 Layer Outputs"])
                
                # Tab 1: Attention Heatmaps
                with internals_tabs[0]:
                    st.markdown("Visualize attention patterns. See which tokens the model focuses on.")
                    
                    # Select Layer and Head
                    col_l, col_h = st.columns(2)
                    # Use stored config
                    viz_cfg = st.session_state.current_cfg
                    with col_l:
                        layer_idx = st.slider("Select Layer", 0, viz_cfg.n_layers - 1, 0)
                    with col_h:
                        head_idx = st.slider("Select Head", 0, viz_cfg.n_heads - 1, 0)
                    
                    # Get attention pattern: [batch, heads, seq, seq] -> [seq, seq]
                    attn_map = diagnostics["attention_patterns"][layer_idx][0, head_idx].cpu().numpy()
                    
                    # Convert tokens to string labels
                    # We might need a better way to decode individual tokens for visualization
                    # For character tokenizer it's easy, for BPE it's harder to get list of strings
                    # We'll make a best effort decoding
                    token_ids = full_tokens[0].cpu().tolist()
                    token_labels = []
                    for tid in token_ids:
                        try:
                            # Attempt to decode single token
                            decoded = st.session_state.current_tokenizer.decode([tid])
                            token_labels.append(f"'{decoded}'")
                        except:
                            token_labels.append(f"T{tid}")
                            
                    
                    # Use shared component (incorporates fix for character-level tokenization)
                    render_attention_heatmap(attn_map, token_labels, layer_idx, head_idx)
                    
                # Tab 2: Logit Lens
                with internals_tabs[1]:
                    st.markdown("""
                    **Logit Lens**: What would the model predict if we stopped after this layer? 
                    We apply the final Unembedding matrix to the output of each intermediate layer.
                    """)
                    
                    # Select a specific position to analyze (default: last token)
                    pos_idx = st.slider("Select Token Position", 0, len(token_labels)-2, len(token_labels)-2,
                                      help="Analyze the prediction for the NEXT token at this position")
                    
                    target_token = token_labels[pos_idx+1]
                    st.write(f"Context: ...{' '.join(token_labels[max(0, pos_idx-5):pos_idx+1])}")
                    st.write(f"**Target (Next Token):** {target_token}")
                    
                    # Collect predictions per layer
                    layer_preds = []
                    
                    # Get Unembed matrix
                    unembed = st.session_state.current_model.unembed
                    
                    for l_i, layer_out in enumerate(diagnostics["layer_outputs"]):
                        # layer_out: [batch, seq, d_model]
                        # Extract vector for position
                        vec = layer_out[0, pos_idx, :] # [d_model]
                        
                        # Apply LN_f (Approximate - we use the model's final LN)
                        # accurate logit lens usually applies final LN then unembed
                        vec_norm = st.session_state.current_model.ln_f(vec.unsqueeze(0).unsqueeze(0)).squeeze()
                        
                        # Project to logits: [d_vocab]
                        if hasattr(unembed, "W_U"): # Manual
                             logits = vec_norm @ unembed.W_U
                        else: # Torch
                             logits = unembed(vec_norm)
                             
                        # Top k
                        probs = torch.softmax(logits, dim=-1)
                        top_vals, top_inds = torch.topk(probs, 5)
                        
                        row = {"Layer": l_i}
                        for k in range(3): # Top 3
                             tok_str = st.session_state.current_tokenizer.decode([top_inds[k].item()])
                             prob = top_vals[k].item()
                             row[f"Rank {k+1}"] = f"'{tok_str}' ({prob:.1%})"
                        layer_preds.append(row)
                    
                    st.table(pd.DataFrame(layer_preds))
                    
                # Tab 3: Layer Outputs (Norms)
                with internals_tabs[2]:
                    # Visualize norm of residual stream
                    st.markdown("Visualizing the 'Growth' of the residual stream (L2 Norm)")
                    norms = []
                    for l_i, layer_out in enumerate(diagnostics["layer_outputs"]):
                        norm = layer_out[0].norm(dim=-1).mean().item()
                        norms.append({"Layer": l_i, "Avg Norm": norm})
                    
                    st.line_chart(pd.DataFrame(norms).set_index("Layer"))
