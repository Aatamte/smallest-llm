"""DataPoint — a single metric data point: step + value."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DataPoint:
    step: int
    value: float

    def to_wire(self) -> dict:
        return {"step": self.step, "value": self.value}
