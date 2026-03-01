"""Seed management for reproducible training."""

import random

import numpy as np
import torch


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_states() -> dict:
    """Capture all RNG states for checkpoint reproducibility."""
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_rng_states(states: dict):
    """Restore RNG states from a checkpoint."""
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.random.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(states["cuda"])
