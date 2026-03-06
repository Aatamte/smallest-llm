"""StepSnapshot — grouped view of all metrics for a single training step."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class StepSnapshot:
    step: int
    train_loss: float | None = None
    lr: float | None = None
    grad_norm: float | None = None
    tokens_seen: int | None = None
    tokens_per_sec: float | None = None
    step_time: float | None = None
    val_loss: float | None = None
    bpc: float | None = None

    def to_wire(self) -> dict:
        d: dict[str, Any] = {"step": self.step}
        if self.train_loss is not None:
            d["trainLoss"] = self.train_loss
        if self.lr is not None:
            d["lr"] = self.lr
        if self.grad_norm is not None:
            d["gradNorm"] = self.grad_norm
        if self.tokens_seen is not None:
            d["tokensSeen"] = self.tokens_seen
        if self.tokens_per_sec is not None:
            d["tokensPerSec"] = self.tokens_per_sec
        if self.step_time is not None:
            d["stepTime"] = self.step_time
        if self.val_loss is not None:
            d["valLoss"] = self.val_loss
        if self.bpc is not None:
            d["bpc"] = self.bpc
        return d

    @classmethod
    def from_metrics(cls, step: int, m: dict[str, float]) -> StepSnapshot:
        train_loss = m.get("train/loss")
        step_time = m.get("train/step_time")
        tokens_seen_raw = m.get("train/tokens_seen")
        tokens_seen = int(tokens_seen_raw) if tokens_seen_raw is not None else None

        tokens_per_sec = None
        if step_time and tokens_seen:
            tokens_per_sec = round(tokens_seen / step_time)

        return cls(
            step=step,
            train_loss=train_loss,
            lr=m.get("train/lr"),
            grad_norm=m.get("train/grad_norm"),
            tokens_seen=tokens_seen,
            tokens_per_sec=tokens_per_sec,
            step_time=round(step_time, 4) if step_time else None,
            val_loss=m.get("val/loss"),
            bpc=round(train_loss / math.log(2), 4) if train_loss else None,
        )
