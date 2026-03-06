"""Layer stats table — per-step per-layer gradient/weight stats."""

from __future__ import annotations

from src.storage.table import Column, Index, Table


class LayerStatsTable(Table):
    name = "layer_stats"
    columns = [
        Column("id", "INTEGER", primary_key=True, autoincrement=True),
        Column("run_id", "INTEGER", not_null=True),
        Column("step", "INTEGER", not_null=True),
        Column("layer", "TEXT", not_null=True),
        Column("grad_norm", "REAL"),
        Column("weight_norm", "REAL"),
        Column("update_ratio", "REAL"),
    ]
    indexes = [
        Index("idx_layer_stats_run_step", ["run_id", "step"]),
    ]

    def log(self, run_id: int, step: int, stats: list[dict]):
        for s in stats:
            self.insert(
                run_id=run_id,
                step=step,
                layer=s["name"],
                grad_norm=s.get("grad_norm"),
                weight_norm=s.get("weight_norm"),
                update_ratio=s.get("update_ratio"),
            )

    def get(self, run_id: int, step: int | None = None) -> list[dict]:
        if step is not None:
            return self.select(
                where="run_id = ? AND step = ?",
                params=[run_id, step],
                order_by="layer",
            )
        return self.select(where="run_id = ?", params=[run_id], order_by="step, layer")
