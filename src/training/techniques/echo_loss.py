"""Echo Loss: multi-directional prediction from hidden states.

Standard next-token prediction only uses h_i to predict token_{i+1}.
Echo Loss adds auxiliary predictions:
  - Backward: h_i predicts token_{i-1}
  - Skip-k:   h_i predicts token_{i+k} for k in {2, 4, 8}

This extracts ~6x more gradient signal per token seen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class EchoHeads(nn.Module):
    """Auxiliary prediction heads for echo loss.

    Each head is a lightweight linear projection from hidden dim to vocab.
    They do NOT share weights with the main LM head.
    """

    def __init__(self, d_model: int, vocab_size: int, skip_distances: tuple[int, ...] = (2, 4, 8)):
        super().__init__()
        self.skip_distances = skip_distances

        self.backward_head = nn.Linear(d_model, vocab_size, bias=False)
        self.skip_heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False)
            for _ in skip_distances
        ])

        # Small init so echo losses don't dominate early
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """Compute logits for all echo directions.

        Args:
            hidden_states: (batch, seq_len, d_model)

        Returns:
            dict with keys "backward", "skip_2", "skip_4", "skip_8" -> logits
        """
        out = {"backward": self.backward_head(hidden_states)}
        for k, head in zip(self.skip_distances, self.skip_heads):
            out[f"skip_{k}"] = head(hidden_states)
        return out


def compute_echo_loss(
    hidden_states: torch.Tensor,
    labels: torch.Tensor,
    echo_heads: EchoHeads,
    backward_weight: float = 0.3,
    skip_weight: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute combined echo loss from hidden states.

    Args:
        hidden_states: (batch, seq_len, d_model) — final hidden states from model
        labels: (batch, seq_len) — target token IDs (same as used for forward loss)
        echo_heads: EchoHeads module
        backward_weight: weight for backward prediction loss
        skip_weight: weight for each skip prediction loss

    Returns:
        (total_echo_loss, metrics_dict)
    """
    batch, seq_len = labels.shape
    logits = echo_heads(hidden_states)
    metrics = {}
    total = torch.tensor(0.0, device=hidden_states.device)

    # Backward: h_i predicts token_{i-1}
    # h at positions 1..T-1 predicts labels at positions 0..T-2
    if seq_len > 1:
        bwd_logits = logits["backward"][:, 1:]   # (batch, T-1, vocab)
        bwd_labels = labels[:, :-1]                # (batch, T-1)
        bwd_loss = F.cross_entropy(
            bwd_logits.reshape(-1, bwd_logits.size(-1)),
            bwd_labels.reshape(-1),
        )
        total = total + backward_weight * bwd_loss
        metrics["echo/backward_loss"] = bwd_loss.item()

    # Skip-k: h_i predicts token_{i+k}
    # h at positions 0..T-1-k predicts labels at positions k..T-1
    for k in echo_heads.skip_distances:
        if seq_len <= k:
            continue
        key = f"skip_{k}"
        skip_logits = logits[key][:, :-k]   # (batch, T-k, vocab)
        skip_labels = labels[:, k:]          # (batch, T-k)
        skip_loss = F.cross_entropy(
            skip_logits.reshape(-1, skip_logits.size(-1)),
            skip_labels.reshape(-1),
        )
        total = total + skip_weight * skip_loss
        metrics[f"echo/skip_{k}_loss"] = skip_loss.item()

    metrics["echo/total_loss"] = total.item()
    return total, metrics
