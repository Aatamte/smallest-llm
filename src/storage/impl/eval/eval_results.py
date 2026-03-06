"""Eval results table — evaluation task results."""

from __future__ import annotations

import json
from typing import Any

from src.storage.table import Column, Index, Table


class EvalResultsTable(Table):
    name = "eval_results"
    columns = [
        Column("id", "INTEGER", primary_key=True, autoincrement=True),
        Column("run_id", "INTEGER"),
        Column("step", "INTEGER"),
        Column("task", "TEXT", not_null=True),
        Column("metrics", "TEXT", not_null=True),
        Column("metadata", "TEXT"),
        Column("model_name", "TEXT"),
        Column("created_at", "TEXT", default="datetime('now')"),
    ]
    indexes = [
        Index("idx_eval_run", ["run_id"]),
        Index("idx_eval_model", ["model_name"]),
        Index("idx_eval_task", ["task"]),
    ]

    def log(
        self,
        task: str,
        metrics: dict,
        metadata: dict | None = None,
        run_id: int | None = None,
        step: int | None = None,
        model_name: str | None = None,
    ):
        self.insert(
            run_id=run_id,
            step=step,
            task=task,
            metrics=json.dumps(metrics),
            metadata=json.dumps(metadata) if metadata else None,
            model_name=model_name,
        )

    def get(
        self,
        run_id: int | None = None,
        task: str | None = None,
        model_name: str | None = None,
    ) -> list[dict]:
        clauses = ["1=1"]
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = ?")
            params.append(run_id)
        if task is not None:
            clauses.append("task = ?")
            params.append(task)
        if model_name is not None:
            clauses.append("model_name = ?")
            params.append(model_name)
        return self.select(
            where=" AND ".join(clauses),
            params=params,
            order_by="created_at",
        )

    def list_models(self) -> list[str]:
        """Return distinct model names that have eval results."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT model_name FROM eval_results "
                "WHERE model_name IS NOT NULL ORDER BY model_name"
            ).fetchall()
            return [r["model_name"] for r in rows]

    def delete_for_run(self, run_id: int):
        self.delete(where="run_id = ?", params=[run_id])
