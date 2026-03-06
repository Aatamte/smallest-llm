"""Checkpoints table — saved model checkpoint records."""

from __future__ import annotations

import json

from src.storage.table import Column, Index, Table


class CheckpointsTable(Table):
    name = "checkpoints"
    columns = [
        Column("id", "INTEGER", primary_key=True, autoincrement=True),
        Column("run_id", "INTEGER", not_null=True, references="runs(id)"),
        Column("step", "INTEGER", not_null=True),
        Column("path", "TEXT", not_null=True),
        Column("metrics", "TEXT"),
        Column("is_best", "INTEGER", default="0"),
        Column("created_at", "TEXT", default="datetime('now')"),
    ]
    indexes = [
        Index("idx_checkpoints_run_step", ["run_id", "step"]),
    ]

    def log(
        self,
        run_id: int,
        step: int,
        path: str,
        metrics: dict | None = None,
        is_best: bool = False,
    ):
        self.insert(
            run_id=run_id,
            step=step,
            path=path,
            metrics=json.dumps(metrics) if metrics else None,
            is_best=1 if is_best else 0,
        )

    def get_for_run(self, run_id: int) -> list[dict]:
        return self.select(where="run_id = ?", params=[run_id], order_by="step")

    def delete_for_run(self, run_id: int):
        self.delete(where="run_id = ?", params=[run_id])
