"""Phantom Batches: synthetic gradient augmentation via hidden state reuse.

After the real forward pass produces hidden states, we apply K random dropout
masks to those hidden states and re-run ONLY the cheap output layers
(final norm + LM head) on each masked version.

Each phantom pass produces a slightly different loss → gradient, effectively
multiplying the batch size by K at ~20% extra compute cost (the expensive
SSM layers are NOT re-run).

This dramatically reduces gradient variance, allowing the optimizer to take
larger, more confident steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PhantomConfig:
    """Configuration for phantom batch generation."""
    n_phantoms: int = 5       # Number of phantom passes per real forward
    dropout_rate: float = 0.1  # Dropout rate for phantom masks
    weight: float = 0.1       # Loss weight per phantom (vs 1.0 for real)


def compute_phantom_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    norm: nn.Module,
    head: nn.Module,
    config: PhantomConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Generate phantom losses from hidden states.

    Args:
        hidden_states: (batch, seq_len, d_model) — from the real forward pass.
                       Must still be in the computation graph (not detached).
        labels: (batch, seq_len) — target token IDs
        norm: The model's final layer norm (e.g., model.norm_f)
        head: The model's output head (e.g., model.head)
        config: PhantomConfig

    Returns:
        (total_phantom_loss, metrics_dict)
    """
    total = torch.tensor(0.0, device=hidden_states.device)
    phantom_losses = []

    for i in range(config.n_phantoms):
        # Apply a unique random dropout mask to the hidden states
        # This perturbs the representations while keeping them on the
        # computation graph so gradients flow back through the SSM
        mask = torch.bernoulli(
            torch.full_like(hidden_states, 1.0 - config.dropout_rate)
        ) / (1.0 - config.dropout_rate)
        masked_hidden = hidden_states * mask

        # Re-run just the cheap output layers
        normed = norm(masked_hidden)
        logits = head(normed)

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        total = total + config.weight * loss
        phantom_losses.append(loss.item())

    metrics = {
        "phantom/mean_loss": sum(phantom_losses) / max(len(phantom_losses), 1),
        "phantom/total_loss": total.item(),
        "phantom/n_phantoms": config.n_phantoms,
    }
    return total, metrics
