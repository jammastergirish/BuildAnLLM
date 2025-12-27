"""
Parameter-Efficient Fine-Tuning (PEFT) methods.

This module provides LoRA (Low-Rank Adaptation) for efficient fine-tuning.
Instead of updating all model parameters, LoRA adds small trainable adapter
matrices to specific layers, dramatically reducing memory and compute requirements.

Main components:
- lora_wrappers.py: Functions to apply LoRA matrices to attention/MLP layers
- lora_utils.py: Model conversion, forward patching, and parameter utilities
"""
