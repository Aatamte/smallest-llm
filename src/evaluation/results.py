"""Evaluation result types."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvalResult:
    """Result from a single evaluation task."""

    task_name: str
    metrics: dict[str, float]
    metadata: dict[str, Any] = field(default_factory=dict)
    per_sample: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        if d["per_sample"] is None:
            del d["per_sample"]
        return d

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    def summary_line(self) -> str:
        """One-line summary for console output."""
        parts = [f"{k}: {v:.4f}" for k, v in self.metrics.items()]
        return f"{self.task_name:>15s} | {' | '.join(parts)}"
