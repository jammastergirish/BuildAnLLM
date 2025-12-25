"""Model loading utilities."""

import torch
from typing import Dict, Any, Tuple
from config import ModelConfig
from pretraining.model.model import TransformerModelWithEinops, TransformerModelWithoutEinops
from pretraining.training.training_args import TransformerTrainingArgs
from utils import print_state_dict_warnings


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[Any, ModelConfig, Dict]:
    """Load model and config from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Tuple of (model, config, checkpoint_dict)
    """
    torch.serialization.add_safe_globals([TransformerTrainingArgs])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = _extract_config(checkpoint)
    model_type = checkpoint.get("model_type", "with_einops")
    model = _create_model(cfg, model_type)
    _load_state_dict(model, checkpoint, device)
    
    model = model.to(device)
    model.eval()
    return model, cfg, checkpoint


def _extract_config(checkpoint: Dict) -> ModelConfig:
    """Extract and reconstruct config from checkpoint."""
    cfg = checkpoint.get("cfg")
    if cfg is None:
        return ModelConfig.gpt_small()
    elif isinstance(cfg, dict):
        return ModelConfig.from_dict(cfg)
    elif isinstance(cfg, ModelConfig):
        return cfg
    else:
        try:
            from dataclasses import asdict
            cfg_dict = asdict(cfg)
            return ModelConfig.from_dict(cfg_dict)
        except Exception:
            return ModelConfig.gpt_small()


def _create_model(cfg: ModelConfig, model_type: str):
    """Create model instance based on type."""
    if model_type == "with_einops":
        return TransformerModelWithEinops(cfg)
    else:
        return TransformerModelWithoutEinops(cfg)


def _load_state_dict(model, checkpoint: Dict, device: torch.device) -> None:
    """Load state dict with shape checking."""
    state_dict = checkpoint["model_state_dict"]
    model_state_dict = dict(model.state_dict())
    
    filtered_state_dict = {}
    missing_keys = []
    unexpected_keys = []

    for key, value in state_dict.items():
        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            else:
                shape_msg = f"{key} (shape mismatch: {value.shape} vs {model_state_dict[key].shape})"
                unexpected_keys.append(shape_msg)
        else:
            unexpected_keys.append(key)

    for key in model_state_dict:
        if key not in filtered_state_dict:
            missing_keys.append(key)

    model.load_state_dict(filtered_state_dict, strict=False)
    print_state_dict_warnings(unexpected_keys, missing_keys)

