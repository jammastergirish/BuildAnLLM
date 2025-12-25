
import re
import threading
from collections import deque
from pathlib import Path

import streamlit as st

from utils import get_device
from pretraining.model.model_loader import load_model_from_checkpoint


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
    st.session_state.shared_training_logs = deque(
        maxlen=200)  # Thread-safe deque
if "training_lock" not in st.session_state:
    st.session_state.training_lock = threading.Lock()


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


with open("README.md", encoding="utf-8") as f:
    content = f.read()
    content = re.sub(r'<img[^>]*>', '', content)
    st.markdown(content)

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
