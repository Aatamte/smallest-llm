"""FlopsEstimate — per-token FLOPs for forward and backward passes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FlopsEstimate:
    forward: int
    backward: int

    @property
    def total(self) -> int:
        return self.forward + self.backward
