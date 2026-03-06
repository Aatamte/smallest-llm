"""LayerStatRecord — per-layer gradient/weight stats."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LayerStatRecord:
    name: str
    grad_norm: float
    weight_norm: float
    update_ratio: float

    def to_wire(self) -> dict:
        return {
            "name": self.name,
            "gradNorm": self.grad_norm,
            "weightNorm": self.weight_norm,
            "updateRatio": self.update_ratio,
        }
