"""Model interface — any model must satisfy this contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    """Standard model output."""
    loss: torch.Tensor
    logits: torch.Tensor


class BaseModel(nn.Module):
    """Interface for all models. Subclass this and implement forward."""

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None, **kwargs
    ) -> ModelOutput:
        raise NotImplementedError

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation with optional top-k sampling."""
        self.eval()
        with torch.no_grad():
            max_ctx = getattr(self, "max_seq_len", None)
            for _ in range(max_new_tokens):
                # Crop to max sequence length if needed
                ctx = input_ids if max_ctx is None else input_ids[:, -max_ctx:]
                logits = self.forward(ctx).logits
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
