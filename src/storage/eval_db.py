"""Separate SQLite database for evaluation results."""

from __future__ import annotations

import json
import sqlite3
from typing import Any


class EvalDatabase:
    """Persistent storage for evaluation results, separate from training data.

    Usage:
        db = EvalDatabase("eval.db")
        db.log_eval(task="perplexity", metrics={"perplexity": 45.2}, model_name="smollm-135m")
        results = db.get_evals(model_name="smollm-135m")
    """

    def __init__(self, path: str = "eval.db"):
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS eval_results (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id     INTEGER,
                step       INTEGER,
                task       TEXT NOT NULL,
                metrics    TEXT NOT NULL,
                metadata   TEXT,
                model_name TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_eval_run
                ON eval_results(run_id);
            CREATE INDEX IF NOT EXISTS idx_eval_model
                ON eval_results(model_name);
            CREATE INDEX IF NOT EXISTS idx_eval_task
                ON eval_results(task);
        """)

    def log_eval(
        self,
        task: str,
        metrics: dict,
        metadata: dict | None = None,
        run_id: int | None = None,
        step: int | None = None,
        model_name: str | None = None,
    ):
        self._conn.execute(
            "INSERT INTO eval_results (run_id, step, task, metrics, metadata, model_name) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                step,
                task,
                json.dumps(metrics),
                json.dumps(metadata) if metadata else None,
                model_name,
            ),
        )
        self._conn.commit()

    def get_evals(
        self,
        run_id: int | None = None,
        task: str | None = None,
        model_name: str | None = None,
    ) -> list[dict]:
        query = "SELECT * FROM eval_results WHERE 1=1"
        params: list[Any] = []
        if run_id is not None:
            query += " AND run_id = ?"
            params.append(run_id)
        if task is not None:
            query += " AND task = ?"
            params.append(task)
        if model_name is not None:
            query += " AND model_name = ?"
            params.append(model_name)
        query += " ORDER BY created_at"
        rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def list_models(self) -> list[str]:
        """Return distinct model names that have eval results."""
        rows = self._conn.execute(
            "SELECT DISTINCT model_name FROM eval_results "
            "WHERE model_name IS NOT NULL ORDER BY model_name"
        ).fetchall()
        return [r["model_name"] for r in rows]

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
