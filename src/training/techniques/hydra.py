"""Hydra Training: multi-resolution parallel forward passes.

Instead of training on one sequence length at a time, runs the SAME model
on multiple resolutions simultaneously in each training step:

  - Micro  (short seqs, large batch): teaches local patterns (bigrams, trigrams)
  - Meso   (medium seqs, medium batch): the main training loop handles this
  - Macro  (long seqs, small batch): teaches long-range coherence

Gradients from all resolutions accumulate before the optimizer step.
SSMs are uniquely suited because cost is O(seq_len), so short sequences
are proportionally cheap.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from src.training.callbacks import CallbackBase

if TYPE_CHECKING:
    from src.training.trainer import Trainer


@dataclass
class HydraResolution:
    """One resolution in the Hydra setup."""
    name: str
    seq_len: int
    batch_size: int
    weight: float  # Loss multiplier


class HydraCallback(CallbackBase):
    """Runs extra forward/backward passes at different sequence resolutions.

    On each step_begin (before the main forward pass), this callback:
    1. Draws a micro batch (short sequences) → forward/backward
    2. Draws a macro batch (long sequences) → forward/backward
    3. Gradients accumulate with the main loop's meso gradients

    The main training loop is unchanged — it just sees extra gradient signal.
    """

    def __init__(
        self,
        micro_loader: DataLoader | None = None,
        macro_loader: DataLoader | None = None,
        micro_weight: float = 0.3,
        macro_weight: float = 0.1,
        weight_schedule: str = "constant",  # "constant" or "shift"
        total_steps: int = 1000,
    ):
        self.micro_loader = micro_loader
        self.macro_loader = macro_loader
        self.micro_weight = micro_weight
        self.macro_weight = macro_weight
        self.weight_schedule = weight_schedule
        self.total_steps = total_steps
        self._micro_iter = None
        self._macro_iter = None

    def _next_batch(self, loader, iter_attr: str) -> dict:
        """Get next batch from an infinite iterator over a loader."""
        it = getattr(self, iter_attr)
        if it is None:
            it = iter(cycle(loader))
            setattr(self, iter_attr, it)
        return next(it)

    def _get_weights(self, step: int) -> tuple[float, float]:
        """Get (micro_weight, macro_weight) for the current step."""
        if self.weight_schedule == "constant":
            return self.micro_weight, self.macro_weight

        # "shift": micro weight decays, macro weight grows
        progress = min(step / max(self.total_steps, 1), 1.0)
        micro_w = self.micro_weight * (1.0 - 0.8 * progress)  # decay to 20%
        macro_w = self.macro_weight * (1.0 + 2.0 * progress)  # grow to 3x
        return micro_w, macro_w

    def _forward_backward(self, trainer: Trainer, batch: dict, weight: float):
        """Run one forward/backward pass with proper AMP handling."""
        batch = {k: v.to(trainer.device) for k, v in batch.items()}

        if trainer._use_amp:
            with torch.autocast(trainer.device.type, dtype=trainer._amp_dtype):
                output = trainer.model(**batch)
                loss = output.loss * weight
            if trainer._scaler is not None and trainer._scaler.is_enabled():
                trainer._scaler.scale(loss).backward()
            else:
                loss.backward()
        else:
            output = trainer.model(**batch)
            loss = output.loss * weight
            loss.backward()

        return loss.item()

    def on_step_begin(self, trainer: Trainer, step: int) -> None:
        """Run micro and macro forward/backward passes before the main loop."""
        micro_w, macro_w = self._get_weights(step)

        trainer.model.train()

        # Micro resolution: short sequences, large batch
        if self.micro_loader is not None and micro_w > 0:
            batch = self._next_batch(self.micro_loader, "_micro_iter")
            self._forward_backward(trainer, batch, micro_w)

        # Macro resolution: long sequences, small batch
        if self.macro_loader is not None and macro_w > 0:
            batch = self._next_batch(self.macro_loader, "_macro_iter")
            self._forward_backward(trainer, batch, macro_w)
