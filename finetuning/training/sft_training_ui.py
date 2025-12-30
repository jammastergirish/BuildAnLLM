"""Training UI components for SFT fine-tuning."""

import streamlit as st
import torch
import threading
from typing import Dict, Any
from collections import deque
from tqdm import tqdm

from finetuning.training.sft_trainer import SFTTrainer


from training_utils import run_training_thread

def train_sft_model_thread(
    trainer: SFTTrainer,
    shared_loss_data: Dict[str, list],
    shared_logs: deque,
    training_active_flag: list,
    lock: threading.Lock,
    progress_data: Dict[str, Any]
) -> None:
    """Wrapper to run the shared training thread logic for SFT."""
    run_training_thread(
        trainer, shared_loss_data, shared_logs, training_active_flag, lock, progress_data
    )
