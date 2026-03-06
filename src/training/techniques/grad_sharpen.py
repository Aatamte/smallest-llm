"""Gradient Sharpening: keep only the top-K% of gradient components.

Muon works by orthogonalizing the momentum buffer, giving better update
directions. Gradient sharpening takes a complementary approach: after
computing the gradient, zero out the weakest components and keep only the
top-K% by magnitude.

Rationale: in a small model, a single batch has useful gradient signal for
only a fraction of parameters. The rest is noise. By zeroing weak components
and rescaling, each update focuses on the parameters that matter most.

Implementation: a callback that runs after backward (on_step_end, before
optimizer.zero_grad) and masks the gradient in-place.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.training.callbacks import CallbackBase

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.training.trainer import Trainer


@dataclass
class GradSharpenConfig:
    """Configuration for gradient sharpening."""
    keep_ratio: float = 0.1  # Keep top 10% of gradient components
    rescale: bool = True       # Rescale kept components to preserve gradient norm


def sharpen_gradients(model: torch.nn.Module, keep_ratio: float = 0.1,
                      rescale: bool = True) -> dict[str, float]:
    """Sharpen gradients in-place: zero weak components, keep top-K%.

    Must be called AFTER backward() and BEFORE optimizer.step().

    Args:
        model: model whose parameter gradients to sharpen
        keep_ratio: fraction of gradient components to keep (0.1 = top 10%)
        rescale: if True, rescale kept components to preserve original norm

    Returns:
        metrics dict with sharpening stats
    """
    # Collect all gradient components into one flat vector
    grad_tensors = []
    grad_shapes = []
    grad_params = []

    for p in model.parameters():
        if p.grad is not None:
            grad_tensors.append(p.grad.flatten())
            grad_shapes.append(p.grad.shape)
            grad_params.append(p)

    if not grad_tensors:
        return {"sharpen/n_kept": 0, "sharpen/n_total": 0}

    flat_grad = torch.cat(grad_tensors)
    n_total = flat_grad.numel()
    n_keep = max(1, int(n_total * keep_ratio))

    # Find threshold: top-K by absolute magnitude
    orig_norm = flat_grad.norm().item()
    abs_grad = flat_grad.abs()
    threshold = torch.topk(abs_grad, n_keep).values[-1]

    # Create mask
    mask = abs_grad >= threshold

    # Apply mask
    if rescale and orig_norm > 0:
        # Compute norm of kept components
        kept_norm = (flat_grad * mask).norm().item()
        scale = orig_norm / max(kept_norm, 1e-10)
    else:
        scale = 1.0

    # Write back sharpened gradients
    offset = 0
    for p, shape in zip(grad_params, grad_shapes):
        numel = p.grad.numel()
        p_mask = mask[offset:offset + numel].reshape(shape)
        p.grad.mul_(p_mask.float())
        if rescale:
            p.grad.mul_(scale)
        offset += numel

    actual_kept = mask.sum().item()
    return {
        "sharpen/n_kept": actual_kept,
        "sharpen/n_total": n_total,
        "sharpen/keep_pct": actual_kept / max(n_total, 1) * 100,
        "sharpen/orig_norm": orig_norm,
        "sharpen/scale": scale,
    }
