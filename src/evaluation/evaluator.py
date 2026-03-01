"""Evaluation protocols and implementations."""

from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import torch
from torch.utils.data import DataLoader


@runtime_checkable
class Evaluator(Protocol):
    name: str

    def evaluate(
        self, model: torch.nn.Module, dataloader: DataLoader, device: torch.device
    ) -> dict[str, float]: ...


class PerplexityEvaluator:
    """Compute perplexity on a dataset."""

    name = "perplexity"

    def evaluate(
        self, model: torch.nn.Module, dataloader: DataLoader, device: torch.device
    ) -> dict[str, float]:
        model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(**batch)
                loss = output["loss"] if isinstance(output, dict) else output.loss
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return {
            "eval/loss": avg_loss,
            "eval/perplexity": math.exp(min(avg_loss, 20)),  # cap to avoid overflow
        }
