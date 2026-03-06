"""CheckpointRecord — mirrors a row in checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass
class CheckpointRecord:
    id: int
    run_id: int
    step: int
    path: str
    metrics: dict[str, float]
    is_best: bool
    created_at: str

    def to_wire(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step": self.step,
            "path": self.path,
            "metrics": self.metrics,
            "is_best": 1 if self.is_best else 0,
            "created_at": self.created_at,
        }

    @classmethod
    def from_db_row(cls, row: dict) -> CheckpointRecord:
        metrics = row.get("metrics", "{}")
        return cls(
            id=row["id"],
            run_id=row["run_id"],
            step=row["step"],
            path=row["path"],
            metrics=json.loads(metrics) if isinstance(metrics, str) else (metrics or {}),
            is_best=bool(row.get("is_best", 0)),
            created_at=row.get("created_at", ""),
        )
