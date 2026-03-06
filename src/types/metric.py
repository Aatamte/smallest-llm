"""MetricRecord — mirrors a row in the metrics table."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MetricRecord:
    run_id: int
    step: int
    key: str
    value: float

    @classmethod
    def from_db_row(cls, row: dict) -> MetricRecord:
        return cls(
            run_id=row["run_id"],
            step=row["step"],
            key=row["key"],
            value=row["value"],
        )
