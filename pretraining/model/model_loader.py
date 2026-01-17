"""Model loading utilities."""

import torch
from typing import Dict, Any, Tuple
from config import ModelConfig
from pretraining.model.model import TransformerModel
from pretraining.training.training_args import TransformerTrainingArgs
from utils import print_state_dict_warnings

# Import FinetuningArgs if available (for loading fine-tuned checkpoints)
try:
    from finetuning.training.finetuning_args import FinetuningArgs
except ImportError:
    FinetuningArgs = None


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> Tuple[Any, ModelConfig, Dict]:
    """Load model and config from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load the model on

    Returns:
        Tuple of (model, config, checkpoint_dict)
    """
    safe_globals = [TransformerTrainingArgs]
    if FinetuningArgs is not None:
        safe_globals.append(FinetuningArgs)
    torch.serialization.add_safe_globals(safe_globals)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    cfg = _extract_config(checkpoint)
    model_type = checkpoint.get("model_type", "with_einops")
    model = _create_model(cfg, model_type)
    
    # Check if this is a LoRA checkpoint
    use_lora = checkpoint.get("use_lora", False)
    if use_lora:
        lora_info = checkpoint.get("lora_info", {})
        # Apply LoRA before loading weights
        try:
            from finetuning.peft.lora_utils import convert_model_to_lora
            model = convert_model_to_lora(
                model,
                rank=lora_info.get("lora_rank", 8),
                alpha=lora_info.get("lora_alpha", 8.0),
                dropout=lora_info.get("lora_dropout", 0.0),
                target_modules=lora_info.get("lora_target_modules", "all"),
            )
        except ImportError:
            print("Warning: LoRA checkpoint detected but LoRA modules not available")
    
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
    use_einops = (model_type == "with_einops")
    return TransformerModel(cfg, use_einops=use_einops)


def _load_state_dict(model, checkpoint: Dict, device: torch.device) -> None:
    """Load state dict with shape checking."""
    state_dict = checkpoint["model_state_dict"]
    model_state_dict = dict(model.state_dict())
    
    filtered_state_dict = {}
    missing_keys = []
    unexpected_keys = []

    # Remap known legacy/TransformerLens keys
    if "embed.W_E" in state_dict and "embed.embedding.weight" not in state_dict:
        state_dict["embed.embedding.weight"] = state_dict["embed.W_E"]
    if "unembed.W_U" in state_dict and "unembed.linear.weight" not in state_dict:
        # Check if transpose is needed
        # W_U is [d_model, d_vocab], linear.weight is [d_vocab, d_model]
        w_u = state_dict["unembed.W_U"]
        target_key = "unembed.linear.weight"
        if target_key in model_state_dict and model_state_dict[target_key].shape != w_u.shape:
             state_dict[target_key] = w_u.T
        else:
             state_dict[target_key] = w_u

    for key, value in state_dict.items():
        if key in model_state_dict:
            if model_state_dict[key].shape == value.shape:
                filtered_state_dict[key] = value
            elif hasattr(value, "T") and model_state_dict[key].shape == value.T.shape:
                 # Auto-transpose if shapes match that way (common for linear layers)
                 filtered_state_dict[key] = value.T
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

