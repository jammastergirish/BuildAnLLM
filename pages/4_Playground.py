"""Playground for experimental tools."""

import streamlit as st
import tiktoken
import torch
from ui_components import render_colored_tokens_rainbow

st.set_page_config(
    page_title="Playground",
    page_icon="🛝",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🛝 Playground")

# --- Sidebar Navigation ---
st.sidebar.title("Tools")
tool_selection = st.sidebar.radio(
    "",
    ["Tokenizer"],
    index=0
)


# --- Tokenizer Tool ---
if tool_selection == "Tokenizer":
    st.header("🔤 Tokenizer")
    st.markdown("Visualize how different models split text into tokens.")
    
    # Model Selector (Top)
    # Get all tiktoken models
    active_models = ["gpt-4", "gpt-3.5-turbo", "gpt-2"]
    all_models = list(tiktoken.model.MODEL_TO_ENCODING.keys())
    # Sort: active first, then alphabetical others
    other_models = sorted([m for m in all_models if m not in active_models])
    model_options = active_models + other_models
    
    col_settings, _ = st.columns([1, 2])
    with col_settings:
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            index=0,
            help="Select the model encoding to use."
        )

    st.divider()
    
    # Main Layout: Side-by-Side
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        st.subheader("⌨️ Input")
        input_text = st.text_area(
            "Input Text",
            value="Hello world! This is a test of the tokenizer.\n\nPython code:\ndef hello():\n    print('world')",
            height=400,
            label_visibility="collapsed",
            help="Type here. Click outside or press Cmd+Enter to update."
        )
        
        # Stats below input
        if input_text:
             st.caption(f"Characters: {len(input_text)}")
    
    with col_output:
        st.subheader("🎨 Tokens")
        
        if input_text:
            # Tokenize
            try:
                encoding = tiktoken.encoding_for_model(selected_model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
                st.warning(f"Could not find exact encoding for {selected_model}, using cl100k_base fallback.")
                
            tokens = encoding.encode(input_text)
            
            # Wrapper for the renderer
            class TiktokenWrapper:
                def __init__(self, encoding):
                    self.encoding = encoding
                
                def decode(self, token_ids):
                    return self.encoding.decode(token_ids)
                    
            wrapped_tokenizer = TiktokenWrapper(encoding)
            
            # Render Colored Tokens
            st.markdown(
                f'<div style="background-color: #262730; padding: 20px; border-radius: 10px; white-space: pre-wrap; font-family: monospace; font-size: 16px; line-height: 1.8; height: 400px; overflow-y: auto;">{render_colored_tokens_rainbow(tokens, wrapped_tokenizer)}</div>',
                unsafe_allow_html=True
            )
            
            # Stats
            sub_c1, sub_c2 = st.columns(2)
            with sub_c1:
                st.metric("Token Count", len(tokens))
            with sub_c2:
                 if len(tokens) > 0:
                    st.metric("Characters per Token", f"{len(input_text)/len(tokens):.2f}")
            
            # Raw IDs
            with st.expander("Raw IDs"):
                st.caption(str(tokens))

