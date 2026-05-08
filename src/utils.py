"""Small helpers shared across the codebase."""
from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Fix random seeds across random / numpy / torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Pick the best available device.

    Order of preference: Apple Silicon (MPS) -> NVIDIA (CUDA) -> CPU.
    Works out of the box on a MacBook Pro M2.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters in a module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
