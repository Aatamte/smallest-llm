"""SQLite database — single source of truth for all experiment data."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from typing import Any


class Database:
    """Persistent storage for training runs, metrics, evals, and checkpoints.

    Thread-safe: all operations are serialized with a lock.

    Usage:
        db = Database("smallest_llm.db")
        run_id = db.create_run("experiment-1", config_dict, env_dict)
        db.log_metrics(run_id, step=100, metrics={"train/loss": 2.3})
        db.log_eval(run_id, step=100, task="perplexity", metrics={"perplexity": 45.2})
        db.finish_run(run_id)
    """

    def __init__(self, path: str = "smallest_llm.db"):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS runs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT NOT NULL,
                config      TEXT NOT NULL,
                env         TEXT,
                status      TEXT DEFAULT 'running',
                created_at  TEXT DEFAULT (datetime('now')),
                finished_at TEXT
            );

            CREATE TABLE IF NOT EXISTS metrics (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id    INTEGER NOT NULL REFERENCES runs(id),
                step      INTEGER NOT NULL,
                key       TEXT NOT NULL,
                value     REAL NOT NULL,
                timestamp TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_metrics_run_step
                ON metrics(run_id, step);
            CREATE INDEX IF NOT EXISTS idx_metrics_run_key
                ON metrics(run_id, key);

            CREATE TABLE IF NOT EXISTS eval_results (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id     INTEGER REFERENCES runs(id),
                step       INTEGER,
                task       TEXT NOT NULL,
                metrics    TEXT NOT NULL,
                metadata   TEXT,
                model_name TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );

        """)

    # ── Runs ──────────────────────────────────────────────

    def create_run(
        self,
        name: str,
        config: dict,
        env: dict | None = None,
    ) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO runs (name, config, env) VALUES (?, ?, ?)",
                (name, json.dumps(config), json.dumps(env) if env else None),
            )
            self._conn.commit()
            return cursor.lastrowid

    def rename_run(self, run_id: int, name: str):
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET name = ? WHERE id = ?", (name, run_id),
            )
            self._conn.commit()

    def finish_run(self, run_id: int, status: str = "completed"):
        with self._lock:
            self._conn.execute(
                "UPDATE runs SET status = ?, finished_at = ? WHERE id = ?",
                (status, datetime.now().isoformat(), run_id),
            )
            self._conn.commit()

    def get_run(self, run_id: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_runs(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, status, created_at, finished_at FROM runs ORDER BY id"
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Metrics ───────────────────────────────────────────

    def log_metrics(self, run_id: int, step: int, metrics: dict[str, float]):
        rows = [
            (run_id, step, k, v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        ]
        if rows:
            with self._lock:
                self._conn.executemany(
                    "INSERT INTO metrics (run_id, step, key, value) VALUES (?, ?, ?, ?)",
                    rows,
                )
                self._conn.commit()

    def get_metrics(
        self,
        run_id: int,
        key: str | None = None,
    ) -> list[dict]:
        with self._lock:
            if key:
                rows = self._conn.execute(
                    "SELECT step, key, value FROM metrics WHERE run_id = ? AND key = ? ORDER BY step",
                    (run_id, key),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT step, key, value FROM metrics WHERE run_id = ? ORDER BY step",
                    (run_id,),
                ).fetchall()
            return [dict(r) for r in rows]

    # ── Eval Results ──────────────────────────────────────

    def log_eval(
        self,
        task: str,
        metrics: dict,
        metadata: dict | None = None,
        run_id: int | None = None,
        step: int | None = None,
        model_name: str | None = None,
    ):
        with self._lock:
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
    ) -> list[dict]:
        query = "SELECT * FROM eval_results WHERE 1=1"
        params: list[Any] = []
        if run_id is not None:
            query += " AND run_id = ?"
            params.append(run_id)
        if task is not None:
            query += " AND task = ?"
            params.append(task)
        query += " ORDER BY created_at"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # ── Delete ────────────────────────────────────────────

    def delete_run(self, run_id: int):
        """Delete a run and all its associated data (metrics, evals, checkpoints)."""
        with self._lock:
            self._conn.execute("DELETE FROM metrics WHERE run_id = ?", (run_id,))
            self._conn.execute("DELETE FROM eval_results WHERE run_id = ?", (run_id,))
            self._conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
            self._conn.commit()

    # ── Recovery ──────────────────────────────────────────

    def mark_stale_runs(self) -> list[int]:
        """Mark any runs still 'running' as 'failed' (server must have died).
        Returns the list of run_ids that were marked."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM runs WHERE status = 'running'"
            ).fetchall()
            stale_ids = [r["id"] for r in rows]
            if stale_ids:
                self._conn.executemany(
                    "UPDATE runs SET status = 'failed', finished_at = datetime('now') WHERE id = ?",
                    [(rid,) for rid in stale_ids],
                )
                self._conn.commit()
            return stale_ids

    # ── Lifecycle ─────────────────────────────────────────

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
