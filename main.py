
import streamlit as st
import torch
import os
import threading
from pathlib import Path

from config import ModelConfig
from training_args import TransformerTrainingArgs


# Page configuration
st.set_page_config(
    page_title="Transformer Training & Inference",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "training_active" not in st.session_state:
    st.session_state.training_active = False
if "trainer" not in st.session_state:
    st.session_state.trainer = None
if "training_thread" not in st.session_state:
    st.session_state.training_thread = None
if "loss_data" not in st.session_state:
    st.session_state.loss_data = {
        "iterations": [], "train_losses": [], "val_losses": []}
if "training_logs" not in st.session_state:
    st.session_state.training_logs = []
if "current_model" not in st.session_state:
    st.session_state.current_model = None
if "current_tokenizer" not in st.session_state:
    st.session_state.current_tokenizer = None
if "shared_loss_data" not in st.session_state:
    st.session_state.shared_loss_data = {
        "iterations": [], "train_losses": [], "val_losses": []}
if "shared_training_logs" not in st.session_state:
    from collections import deque
    st.session_state.shared_training_logs = deque(
        maxlen=200)  # Thread-safe deque
if "training_lock" not in st.session_state:
    st.session_state.training_lock = threading.Lock()


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def scan_checkpoints():
    """Scan checkpoints directory and return available checkpoints."""
    checkpoints_dir = Path("checkpoints")
    if not checkpoints_dir.exists():
        return []

    checkpoints = []
    for checkpoint_dir in sorted(checkpoints_dir.iterdir(), reverse=True):
        if checkpoint_dir.is_dir():
            # Look for final_model.pt first, then any checkpoint_*.pt
            final_model = checkpoint_dir / "final_model.pt"
            if final_model.exists():
                checkpoints.append({
                    "path": str(final_model),
                    "name": f"{checkpoint_dir.name} (final)",
                    "timestamp": checkpoint_dir.name
                })
            else:
                # Get all checkpoint files
                for ckpt_file in sorted(checkpoint_dir.glob("checkpoint_*.pt"), reverse=True):
                    checkpoints.append({
                        "path": str(ckpt_file),
                        "name": f"{checkpoint_dir.name} / {ckpt_file.stem}",
                        "timestamp": checkpoint_dir.name
                    })

    return checkpoints


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model and config from checkpoint."""
    from training_args import TransformerTrainingArgs
    torch.serialization.add_safe_globals([TransformerTrainingArgs])
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False)

    cfg = checkpoint.get("cfg")
    if cfg is None:
        cfg = ModelConfig.gpt_small()
    elif isinstance(cfg, dict):
        cfg = ModelConfig(**cfg)

    model_type = checkpoint.get("model_type", "with_einops")

    if model_type == "with_einops":
        from model import TransformerModelWithEinops
        model = TransformerModelWithEinops(cfg)
    else:
        from model import TransformerModelWithoutEinops
        model = TransformerModelWithoutEinops(cfg)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, cfg, checkpoint


# Main page - just show welcome message
st.title("🤖 Transformer Training & Inference")
st.markdown("""
Welcome to the Transformer Training & Inference App!

Use the sidebar to navigate to:
- **🚂 Training**: Train transformer models (GPT, LLaMA, or OLMo) with live visualization
- **🎯 Inference**: Generate text from trained models

### Features
- Upload your own training data or use the default file
- Choose between GPT, LLaMA, and OLMo architectures
- Real-time training progress with live loss graphs
- Interactive text generation with customizable sampling parameters
""")

# Store helper functions in session state for access by pages
if "get_device" not in st.session_state:
    st.session_state.get_device = get_device
if "scan_checkpoints" not in st.session_state:
    st.session_state.scan_checkpoints = scan_checkpoints
if "load_model_from_checkpoint" not in st.session_state:
    st.session_state.load_model_from_checkpoint = load_model_from_checkpoint

# Note: Training and Inference pages are in the pages/ directory
# Streamlit automatically creates navigation for files in pages/
# - pages/1_Training.py: Training page
# - pages/2_Inference.py: Inference page
