"""Activation stats table — per-step per-layer activation stats."""

from __future__ import annotations

from src.storage.table import Column, Index, Table


class ActivationStatsTable(Table):
    name = "activation_stats"
    columns = [
        Column("id", "INTEGER", primary_key=True, autoincrement=True),
        Column("run_id", "INTEGER", not_null=True),
        Column("step", "INTEGER", not_null=True),
        Column("layer", "TEXT", not_null=True),
        Column("mean", "REAL"),
        Column("std", "REAL"),
        Column("min_val", "REAL"),
        Column("max_val", "REAL"),
        Column("pct_zero", "REAL"),
    ]
    indexes = [
        Index("idx_activation_stats_run_step", ["run_id", "step"]),
    ]

    def log(self, run_id: int, step: int, stats: list[dict]):
        for s in stats:
            self.insert(
                run_id=run_id,
                step=step,
                layer=s["name"],
                mean=s.get("mean"),
                std=s.get("std"),
                min_val=s.get("min"),
                max_val=s.get("max"),
                pct_zero=s.get("pct_zero"),
            )

    def get(self, run_id: int, step: int | None = None) -> list[dict]:
        if step is not None:
            return self.select(
                where="run_id = ? AND step = ?",
                params=[run_id, step],
                order_by="layer",
            )
        return self.select(where="run_id = ?", params=[run_id], order_by="step, layer")
