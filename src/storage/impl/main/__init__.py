"""MainDatabase — smallest_llm.db with runs, metrics, checkpoints, models, and live data."""

from __future__ import annotations

from src.storage.database import Database
from src.storage.impl.main.checkpoints import CheckpointsTable
from src.storage.impl.main.generations import GenerationsTable
from src.storage.impl.main.logs import LogsTable
from src.storage.impl.main.metrics import MetricsTable
from src.storage.impl.main.models import ModelsTable
from src.storage.impl.main.runs import RunsTable


class MainDatabase(Database):
    """Primary database for training runs and associated data.

    Tables: runs, metrics, checkpoints, models, generations, logs.
    """

    def __init__(self, path: str = "smallest_llm.db"):
        super().__init__(path)

        self.runs = RunsTable(self.conn, self.lock)
        self.metrics = MetricsTable(self.conn, self.lock)
        self.checkpoints = CheckpointsTable(self.conn, self.lock)
        self.models = ModelsTable(self.conn, self.lock)
        self.generations = GenerationsTable(self.conn, self.lock)
        self.logs = LogsTable(self.conn, self.lock)

        self._register(self.runs)
        self._register(self.metrics)
        self._register(self.checkpoints)
        self._register(self.models)
        self._register(self.generations)
        self._register(self.logs)
        self._create_all()
        self._migrate()

    # ── Convenience delegations (preserve existing API) ────

    def create_run(self, name: str, config: dict, env: dict | None = None) -> int:
        return self.runs.create_run(name, config, env)

    def rename_run(self, run_id: int, name: str):
        self.runs.rename_run(run_id, name)

    def finish_run(self, run_id: int, status: str = "completed"):
        self.runs.finish_run(run_id, status)

    def get_run(self, run_id: int) -> dict | None:
        return self.runs.get_run(run_id)

    def list_runs(self) -> list[dict]:
        return self.runs.list_runs()

    def log_metrics(self, run_id: int, step: int, metrics: dict[str, float]):
        self.metrics.log(run_id, step, metrics)

    def get_metrics(self, run_id: int, key: str | None = None) -> list[dict]:
        return self.metrics.get(run_id, key)

    def log_checkpoint(self, run_id: int, step: int, path: str, metrics: dict | None = None, is_best: bool = False):
        self.checkpoints.log(run_id, step, path, metrics, is_best)

    def get_checkpoints(self, run_id: int) -> list[dict]:
        return self.checkpoints.get_for_run(run_id)

    def create_model(self, run_id: int | None, name: str, path: str, config: dict | None = None) -> int:
        return self.models.create_model(run_id, name, path, config)

    def list_models(self) -> list[dict]:
        return self.models.list_all()

    def get_model(self, model_id: int) -> dict | None:
        return self.models.get(model_id)

    def get_model_by_name(self, name: str) -> dict | None:
        return self.models.get_by_name(name)

    def delete_model(self, model_id: int):
        self.models.delete_model(model_id)

    def sync_models_dir(self, models_dir: str = "models"):
        self.models.sync_dir(models_dir)

    def mark_stale_runs(self) -> list[int]:
        return self.runs.mark_stale()

    def set_live_status(self, run_id: int, status: str):
        self.runs.set_live_status(run_id, status)

    def set_text_state(self, run_id: int, text_state: str):
        self.runs.set_text_state(run_id, text_state)

    def set_stage(
        self,
        run_id: int,
        stage_index: int,
        stage_name: str,
        total_stages: int,
        dataset: str | None = None,
        stage_type: str = "pretrain",
    ):
        self.runs.set_stage(run_id, stage_index, stage_name, total_stages, dataset, stage_type)

    def delete_run(self, run_id: int):
        """Delete a run and all associated data."""
        self.metrics.delete(where="run_id = ?", params=[run_id])
        self.checkpoints.delete_for_run(run_id)
        self.runs.delete(where="id = ?", params=[run_id])

    # ── Migration ─────────────────────────────────────────

    def _migrate(self):
        """Add columns from run_state merge and drop deleted tables. Idempotent."""
        new_cols = [
            ("live_status", "TEXT DEFAULT 'idle'"),
            ("text_state", "TEXT DEFAULT ''"),
            ("stage_index", "INTEGER DEFAULT 0"),
            ("stage_name", "TEXT DEFAULT ''"),
            ("total_stages", "INTEGER DEFAULT 0"),
            ("dataset", "TEXT"),
            ("stage_type", "TEXT DEFAULT 'pretrain'"),
        ]
        with self.lock:
            existing = {row[1] for row in self.conn.execute("PRAGMA table_info(runs)").fetchall()}
            for col_name, col_type in new_cols:
                if col_name not in existing:
                    self.conn.execute(f"ALTER TABLE runs ADD COLUMN {col_name} {col_type}")
            self.conn.execute("DROP TABLE IF EXISTS layer_stats")
            self.conn.execute("DROP TABLE IF EXISTS activation_stats")
            self.conn.execute("DROP TABLE IF EXISTS run_state")
            self.conn.commit()
