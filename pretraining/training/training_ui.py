"""Training UI components and thread management."""

import streamlit as st
import torch
import threading
from typing import Dict, Any, List
from collections import deque

from pretraining.training.trainer import TransformerTrainer
from pretraining.utils import extract_model_output_and_aux_loss, add_aux_loss_to_main_loss


def initialize_training_state():
    """Initialize training-related session state."""
    if "training_active" not in st.session_state:
        st.session_state.training_active = False
    if "trainer" not in st.session_state:
        st.session_state.trainer = None
    if "training_thread" not in st.session_state:
        st.session_state.training_thread = None
    if "shared_loss_data" not in st.session_state:
        st.session_state.shared_loss_data = {
            "iterations": [], "train_losses": [], "val_losses": []
        }
    if "shared_training_logs" not in st.session_state:
        st.session_state.shared_training_logs = deque(maxlen=200)
    if "training_lock" not in st.session_state:
        st.session_state.training_lock = threading.Lock()
    if "loss_data" not in st.session_state:
        st.session_state.loss_data = {
            "iterations": [], "train_losses": [], "val_losses": []
        }
    if "training_logs" not in st.session_state:
        st.session_state.training_logs = []


from training_utils import run_training_thread

def train_model_thread(
    trainer: TransformerTrainer,
    shared_loss_data: Dict[str, List],
    shared_logs: deque,
    training_active_flag: List[bool],
    lock: threading.Lock,
    progress_data: Dict[str, Any]
) -> None:
    """Wrapper to run the shared training thread logic."""
    run_training_thread(
        trainer, shared_loss_data, shared_logs, training_active_flag, lock, progress_data
    )
