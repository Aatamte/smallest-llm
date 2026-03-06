"""Per-layer reconstruction loss: penalize information destruction.

Each layer in a residual network computes x → x + f(x). The residual
connection preserves the input, but f(x) can still corrupt or overwrite
information. This module adds an auxiliary loss that encourages each
layer's output to be linearly invertible — i.e., you can recover the
input from the output.

Empirically, training already pushes layers toward higher invertibility
(R² increases from ~61% to ~83% on Mamba after 200 steps), but the
architecture fights this — especially in SSMs where the sequential scan
has no internal residual. This loss helps the model do what it's already
trying to do.

Implementation: for each adjacent layer pair, project the output back to
the input space via a small learned linear map and minimize reconstruction
MSE. The projections are cheap (D×D matrices) and only used during training.

Usage:
    recon = LayerReconstruction(d_model=128, n_layers=4, weight=0.01)
    # In training loop, after forward pass:
    recon_loss, metrics = recon(layer_activations)
    total_loss = task_loss + recon_loss
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerReconstruction(nn.Module):
    """Per-layer reconstruction loss.

    For each layer transition i → i+1, learns a linear map W_i that
    reconstructs the input: input_hat = W_i @ output + b_i.
    Loss = mean MSE across all layers.

    The reconstruction maps are lightweight (D×D + D per layer) and
    only participate in the backward pass — they don't affect the
    model's forward computation.
    """

    def __init__(self, d_model: int, n_layers: int, weight: float = 0.01):
        super().__init__()
        self.weight = weight
        self.n_layers = n_layers

        # One reconstruction map per layer transition
        self.recon_maps = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_layers)
        ])

        # Small init — reconstruction loss shouldn't dominate early
        for m in self.recon_maps:
            # Init close to identity so reconstruction starts good
            nn.init.eye_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(
        self, activations: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute reconstruction loss from layer activations.

        Args:
            activations: list of (batch, seq_len, d_model) tensors,
                         length n_layers + 1 (embedding + each layer output)

        Returns:
            (weighted_loss, metrics_dict)
        """
        assert len(activations) == self.n_layers + 1, (
            f"Expected {self.n_layers + 1} activations, got {len(activations)}"
        )

        total_loss = torch.tensor(0.0, device=activations[0].device)
        metrics: dict[str, float] = {}

        for i in range(self.n_layers):
            inp = activations[i].detach()  # don't backprop through target
            out = activations[i + 1]       # DO backprop through this

            # Reconstruct input from output
            inp_hat = self.recon_maps[i](out)

            # Normalized MSE (scale-invariant)
            mse = (inp - inp_hat).pow(2).mean()
            inp_var = inp.var() + 1e-8
            normalized_mse = mse / inp_var

            total_loss = total_loss + normalized_mse
            metrics[f"recon/layer_{i}_mse"] = mse.item()
            metrics[f"recon/layer_{i}_r2"] = (1.0 - normalized_mse.item())

        avg_loss = total_loss / self.n_layers
        weighted = self.weight * avg_loss

        metrics["recon/avg_loss"] = avg_loss.item()
        metrics["recon/weighted_loss"] = weighted.item()

        return weighted, metrics


def collect_layer_activations(
    model: nn.Module, input_ids: torch.Tensor,
) -> list[torch.Tensor]:
    """Run a forward pass and collect activations at each layer.

    Works with any model that has:
      - .token_emb (embedding layer)
      - .layers (ModuleList of layers)
      - .drop (optional dropout)

    Returns list of tensors: [embedding_output, layer_0_output, ..., layer_N_output]
    """
    activations = []

    # Embedding
    x = model.token_emb(input_ids)
    if hasattr(model, "drop"):
        x = model.drop(x)
    activations.append(x)

    # Each layer
    for layer in model.layers:
        x = layer(x)
        activations.append(x)

    return activations
