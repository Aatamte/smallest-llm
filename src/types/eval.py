"""EvalRecord — mirrors a row in eval_results."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass
class EvalRecord:
    id: int
    run_id: int | None
    step: int | None
    task: str
    metrics: dict[str, float]
    metadata: dict[str, Any]
    model_name: str | None
    created_at: str

    def to_wire(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "step": self.step,
            "task": self.task,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "model_name": self.model_name,
            "created_at": self.created_at,
        }

    @classmethod
    def from_db_row(cls, row: dict) -> EvalRecord:
        metrics = row.get("metrics", "{}")
        metadata = row.get("metadata")
        return cls(
            id=row["id"],
            run_id=row.get("run_id"),
            step=row.get("step"),
            task=row["task"],
            metrics=json.loads(metrics) if isinstance(metrics, str) else (metrics or {}),
            metadata=json.loads(metadata) if isinstance(metadata, str) else (metadata or {}),
            model_name=row.get("model_name"),
            created_at=row.get("created_at", ""),
        )
