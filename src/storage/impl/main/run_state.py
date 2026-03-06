"""Run state table — mutable per-run state (status, stage, etc.)."""

from __future__ import annotations

from src.storage.table import Column, Table


class RunStateTable(Table):
    name = "run_state"
    columns = [
        Column("run_id", "INTEGER", primary_key=True),
        Column("status", "TEXT", default="'idle'"),
        Column("text_state", "TEXT", default="''"),
        Column("stage_index", "INTEGER", default="0"),
        Column("stage_name", "TEXT", default="''"),
        Column("total_stages", "INTEGER", default="0"),
        Column("dataset", "TEXT"),
        Column("stage_type", "TEXT", default="'pretrain'"),
        Column("max_steps", "INTEGER", default="0"),
        Column("start_time", "TEXT"),
        Column("tokens_per_step", "INTEGER", default="0"),
    ]
    indexes = []

    def set_status(self, run_id: int, status: str):
        self.upsert(run_id=run_id, status=status)

    def set_text_state(self, run_id: int, text_state: str):
        self.upsert(run_id=run_id, text_state=text_state)

    def set_stage(
        self,
        run_id: int,
        stage_index: int,
        stage_name: str,
        total_stages: int,
        dataset: str | None = None,
        stage_type: str = "pretrain",
    ):
        self.upsert(
            run_id=run_id,
            stage_index=stage_index,
            stage_name=stage_name,
            total_stages=total_stages,
            dataset=dataset,
            stage_type=stage_type,
        )

    def get_state(self, run_id: int) -> dict | None:
        rows = self.select(where="run_id = ?", params=[run_id])
        return rows[0] if rows else None
