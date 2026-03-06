"""EvalDatabase — eval.db with eval_results."""

from __future__ import annotations

from src.storage.database import Database
from src.storage.impl.eval.eval_results import EvalResultsTable


class EvalDatabase(Database):
    """Evaluation results database.

    Table: eval_results.
    """

    def __init__(self, path: str = "eval.db"):
        super().__init__(path)

        self.eval_results = EvalResultsTable(self.conn, self.lock)

        self._register(self.eval_results)
        self._create_all()

    # ── Convenience delegations ────

    def log_eval(
        self,
        task: str,
        metrics: dict,
        metadata: dict | None = None,
        run_id: int | None = None,
        step: int | None = None,
        model_name: str | None = None,
    ):
        self.eval_results.log(task, metrics, metadata, run_id, step, model_name)

    def get_evals(
        self,
        run_id: int | None = None,
        task: str | None = None,
        model_name: str | None = None,
    ) -> list[dict]:
        return self.eval_results.get(run_id, task, model_name)

    def list_models(self) -> list[str]:
        return self.eval_results.list_models()

    def delete_by_run(self, run_id: int):
        self.eval_results.delete_for_run(run_id)
