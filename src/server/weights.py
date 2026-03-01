"""Extract and downsample model weights from checkpoints for visualization."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def extract_weights(checkpoint_path: str, max_size: int = 64) -> list[dict]:
    """Load a checkpoint and return downsampled weight tensors.

    Returns list of {name, shape, data} where data is a 2D list of floats.
    """
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_state = state.get("model_state_dict", {})

    layers = []
    for name, tensor in model_state.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        original_shape = list(tensor.shape)

        # Skip scalars
        if tensor.ndim == 0:
            continue

        # Convert to 2D for uniform handling
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim > 2:
            # Flatten extra dims into rows (e.g. conv1d [out, 1, k] -> [out, k])
            tensor = tensor.reshape(tensor.shape[0], -1)

        # Downsample if too large
        h, w = tensor.shape
        if h > max_size or w > max_size:
            tensor = tensor.float().unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            new_h = min(h, max_size)
            new_w = min(w, max_size)
            tensor = F.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)
            tensor = tensor.squeeze(0).squeeze(0)

        # Round to reduce JSON size
        data = [[round(v, 5) for v in row] for row in tensor.tolist()]

        layers.append({
            "name": name,
            "shape": original_shape,
            "data": data,
        })

    return layers
