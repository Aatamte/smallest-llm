"""Metrics table — per-step training metrics."""

from __future__ import annotations

import math

from src.storage.table import Column, Index, Table


class MetricsTable(Table):
    name = "metrics"
    columns = [
        Column("id", "INTEGER", primary_key=True, autoincrement=True),
        Column("run_id", "INTEGER", not_null=True, references="runs(id)"),
        Column("step", "INTEGER", not_null=True),
        Column("key", "TEXT", not_null=True),
        Column("value", "REAL", not_null=True),
        Column("timestamp", "TEXT", default="datetime('now')"),
    ]
    indexes = [
        Index("idx_metrics_run_step", ["run_id", "step"]),
        Index("idx_metrics_run_key", ["run_id", "key"]),
    ]

    def log(self, run_id: int, step: int, metrics: dict[str, float]):
        rows = []
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v):
                rows.append({"run_id": run_id, "step": step, "key": k, "value": v})
        if rows:
            self.insert_many(rows)

    def get(self, run_id: int, key: str | None = None) -> list[dict]:
        if key:
            return self.select(
                columns="step, key, value",
                where="run_id = ? AND key = ?",
                params=[run_id, key],
                order_by="step",
            )
        return self.select(
            columns="step, key, value",
            where="run_id = ?",
            params=[run_id],
            order_by="step",
        )
