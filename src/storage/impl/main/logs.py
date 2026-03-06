"""Logs table — log lines from training."""

from __future__ import annotations

from src.storage.table import Column, Index, Table


class LogsTable(Table):
    name = "logs"
    columns = [
        Column("id", "INTEGER", primary_key=True, autoincrement=True),
        Column("run_id", "INTEGER"),
        Column("level", "TEXT", not_null=True),
        Column("message", "TEXT", not_null=True),
        Column("created_at", "TEXT", default="datetime('now')"),
    ]
    indexes = [
        Index("idx_logs_run", ["run_id"]),
    ]

    def log(self, run_id: int | None, level: str, message: str):
        self.insert(run_id=run_id, level=level, message=message)

    def get(self, run_id: int | None = None) -> list[dict]:
        if run_id is not None:
            return self.select(where="run_id = ?", params=[run_id], order_by="id")
        return self.select(order_by="id")
