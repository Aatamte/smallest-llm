"""Device resolution and MPS workarounds."""

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve device string to torch.device. Priority: MPS > CUDA > CPU."""
    if requested != "auto":
        return torch.device(requested)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def synchronize(device: torch.device):
    """Device-agnostic synchronize for accurate timing."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def empty_cache(device: torch.device):
    """Free unused cached memory."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
