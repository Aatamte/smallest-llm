"""Runs table — training run records."""

from __future__ import annotations

import json
from datetime import datetime

from src.storage.table import Column, Index, Table


class RunsTable(Table):
    name = "runs"
    columns = [
        Column("id", "INTEGER", primary_key=True, autoincrement=True),
        Column("name", "TEXT", not_null=True),
        Column("config", "TEXT", not_null=True),
        Column("env", "TEXT"),
        Column("status", "TEXT", default="'running'"),
        Column("created_at", "TEXT", default="datetime('now')"),
        Column("finished_at", "TEXT"),
    ]
    indexes = []

    def create_run(self, name: str, config: dict, env: dict | None = None) -> int:
        return self.insert(
            name=name,
            config=json.dumps(config),
            env=json.dumps(env) if env else None,
        )

    def rename_run(self, run_id: int, name: str):
        self.update("name = ?", "id = ?", [name, run_id])

    def finish_run(self, run_id: int, status: str = "completed"):
        self.update(
            "status = ?, finished_at = ?",
            "id = ?",
            [status, datetime.now().isoformat(), run_id],
        )

    def get_run(self, run_id: int) -> dict | None:
        rows = self.select(where="id = ?", params=[run_id])
        return rows[0] if rows else None

    def list_runs(self) -> list[dict]:
        return self.select(
            columns="id, name, status, created_at, finished_at",
            order_by="id",
        )

    def mark_stale(self) -> list[int]:
        """Mark runs still 'running' as 'failed'. Returns affected run_ids."""
        rows = self.select(where="status = 'running'", columns="id")
        stale_ids = [r["id"] for r in rows]
        for rid in stale_ids:
            self.update(
                "status = ?, finished_at = datetime('now')",
                "id = ?",
                ["failed", rid],
            )
        return stale_ids
