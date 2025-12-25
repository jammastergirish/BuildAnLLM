"""Shared utility functions for training and inference."""

import torch
from typing import List


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_state_dict_warnings(unexpected_keys: List[str], missing_keys: List[str]) -> None:
    """Print warnings about state dict mismatches."""
    if unexpected_keys:
        print(f"Warning: {len(unexpected_keys)} unexpected key(s) in checkpoint (ignored):")
        for key in unexpected_keys[:5]:
            print(f"  - {key}")
        if len(unexpected_keys) > 5:
            print(f"  ... and {len(unexpected_keys) - 5} more")

    if missing_keys:
        print(f"Warning: {len(missing_keys)} missing key(s) in checkpoint (using random initialization):")
        for key in missing_keys[:5]:
            print(f"  - {key}")
        if len(missing_keys) > 5:
            print(f"  ... and {len(missing_keys) - 5} more")

