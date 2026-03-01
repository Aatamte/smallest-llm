"""Separate SQLite database for checkpoint records."""

from __future__ import annotations

import json
import sqlite3
import threading


class CheckpointDatabase:
    """Persistent storage for checkpoint records, separate from training data.

    Usage:
        db = CheckpointDatabase("checkpoints.db")
        db.log_checkpoint(run_id=1, step=100, path="/checkpoints/step-100.pt")
        checkpoints = db.get_checkpoints(run_id=1)
    """

    def __init__(self, path: str = "checkpoints.db"):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoints (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id     INTEGER NOT NULL,
                step       INTEGER NOT NULL,
                path       TEXT NOT NULL,
                metrics    TEXT,
                is_best    INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_checkpoints_run_step
                ON checkpoints(run_id, step);
        """)

    def log_checkpoint(
        self,
        run_id: int,
        step: int,
        path: str,
        metrics: dict | None = None,
        is_best: bool = False,
    ):
        with self._lock:
            self._conn.execute(
                "INSERT INTO checkpoints (run_id, step, path, metrics, is_best) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    run_id,
                    step,
                    path,
                    json.dumps(metrics) if metrics else None,
                    1 if is_best else 0,
                ),
            )
            self._conn.commit()

    def get_checkpoints(self, run_id: int) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM checkpoints WHERE run_id = ? ORDER BY step",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_by_run(self, run_id: int):
        """Delete all checkpoints for a given run_id."""
        with self._lock:
            self._conn.execute("DELETE FROM checkpoints WHERE run_id = ?", (run_id,))
            self._conn.commit()

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
