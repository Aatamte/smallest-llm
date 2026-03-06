"""Generations table — sample text generations during training."""

from __future__ import annotations

from src.storage.table import Column, Index, Table


class GenerationsTable(Table):
    name = "generations"
    columns = [
        Column("id", "INTEGER", primary_key=True, autoincrement=True),
        Column("run_id", "INTEGER", not_null=True),
        Column("step", "INTEGER", not_null=True),
        Column("prompt", "TEXT"),
        Column("output", "TEXT"),
        Column("created_at", "TEXT", default="datetime('now')"),
    ]
    indexes = [
        Index("idx_generations_run", ["run_id"]),
    ]

    def log(self, run_id: int, step: int, prompt: str, output: str):
        self.insert(run_id=run_id, step=step, prompt=prompt, output=output)

    def get(self, run_id: int) -> list[dict]:
        return self.select(where="run_id = ?", params=[run_id], order_by="step")
