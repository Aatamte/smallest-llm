"""State Anchoring Loss: penalize information decay in SSM hidden states.

Standard next-token loss never explicitly tells the SSM to *remember*.
If the model can predict token t+1 without remembering token t-8, it will
forget token t-8. State anchoring adds a loss that forces the hidden state
at position t to retain information about positions t-k.

Implementation: for each anchor distance k, project hidden states at position
t and t-k through small learned projections, then maximize their cosine
similarity. This teaches the SSM to build durable representations.

Cost: ~5-10% extra compute per step (just linear projections + cosine sim).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StateAnchorConfig:
    """Configuration for state anchoring loss."""
    distances: tuple[int, ...] = (4, 8, 16)  # Anchor distances k
    weight: float = 0.1                        # Loss weight
    proj_dim: int = 32                          # Projection dimension


class StateAnchorHeads(nn.Module):
    """Learned projections for state anchoring.

    For each anchor distance k, we have two projections:
      - query_proj: maps h[t] to a query vector
      - key_proj: maps h[t-k] to a key vector

    The loss maximizes cosine similarity between query and key.
    Separate projections per distance allow the model to learn
    what information matters at different time scales.
    """

    def __init__(self, d_model: int, config: StateAnchorConfig):
        super().__init__()
        self.config = config

        self.query_projs = nn.ModuleDict()
        self.key_projs = nn.ModuleDict()

        for k in config.distances:
            self.query_projs[str(k)] = nn.Linear(d_model, config.proj_dim, bias=False)
            self.key_projs[str(k)] = nn.Linear(d_model, config.proj_dim, bias=False)

        # Small init so anchor loss doesn't dominate early
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute query and key projections for all distances.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            dict with keys like "q_4", "k_4", "q_8", "k_8", etc.
        """
        out = {}
        for k in self.config.distances:
            ks = str(k)
            out[f"q_{k}"] = self.query_projs[ks](hidden_states)
            out[f"k_{k}"] = self.key_projs[ks](hidden_states)
        return out


def compute_state_anchor_loss(
    hidden_states: torch.Tensor,
    anchor_heads: StateAnchorHeads,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute state anchoring loss.

    For each distance k:
      query = proj_q(h[t])    for t in [k, seq_len)
      key   = proj_k(h[t-k])  for t in [k, seq_len)
      loss  = 1 - mean(cosine_similarity(query, key))

    Args:
        hidden_states: (batch, seq_len, d_model)
        anchor_heads: StateAnchorHeads module

    Returns:
        (total_loss, metrics_dict)
    """
    config = anchor_heads.config
    seq_len = hidden_states.shape[1]
    projections = anchor_heads(hidden_states)
    metrics = {}
    total = torch.tensor(0.0, device=hidden_states.device)

    for k in config.distances:
        if k >= seq_len:
            continue

        query = projections[f"q_{k}"][:, k:]       # (batch, seq-k, proj_dim)
        key = projections[f"k_{k}"][:, :seq_len-k]  # (batch, seq-k, proj_dim)

        # Normalize for cosine similarity
        query = F.normalize(query, dim=-1)
        key = F.normalize(key, dim=-1)

        # Cosine similarity: higher = better retention
        cos_sim = (query * key).sum(dim=-1)  # (batch, seq-k)
        # Loss = 1 - mean similarity (we want to maximize similarity)
        anchor_loss = 1.0 - cos_sim.mean()

        total = total + config.weight * anchor_loss
        metrics[f"anchor/k={k}_sim"] = cos_sim.mean().item()
        metrics[f"anchor/k={k}_loss"] = anchor_loss.item()

    metrics["anchor/total_loss"] = total.item()
    return total, metrics
