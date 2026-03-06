"""ActivationStatRecord — per-layer activation stats."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ActivationStatRecord:
    name: str
    mean: float
    std: float
    max: float
    min: float
    pct_zero: float

    def to_wire(self) -> dict:
        return {
            "name": self.name,
            "mean": self.mean,
            "std": self.std,
            "max": self.max,
            "min": self.min,
            "pctZero": self.pct_zero,
        }
