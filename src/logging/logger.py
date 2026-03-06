"""Logging: SQLite tables + console. CDC ops are emitted automatically by Table."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from src.types.activation_stat import ActivationStatRecord
from src.types.layer_stat import LayerStatRecord

if TYPE_CHECKING:
    from src.config.base import LoggingConfig
    from src.storage.impl.main import MainDatabase
    from src.types.status import WireStatus


class Logger:
    def __init__(
        self,
        config: LoggingConfig,
        experiment_name: str,
        db: MainDatabase | None = None,
        run_id: int | None = None,
        broadcaster=None,  # kept for backward compat, unused
    ):
        self.config = config
        self._db = db
        self._run_id = run_id
        self._step_count = 0

    def log_step(self, metrics: dict, step: int | None = None):
        """Log metrics to SQLite. CDC ops are emitted automatically."""
        if step is None:
            step = metrics.get("train/step", self._step_count)
        self._step_count = step

        if self._db is not None and self._run_id is not None:
            self._db.log_metrics(self._run_id, step, metrics)

        # Console
        if step % self.config.console_interval == 0:
            self._print_metrics(metrics, step)

    def broadcast_layers(self, layer_stats: list[LayerStatRecord]):
        """Write per-layer gradient/weight stats to DB."""
        if self._db is not None and self._run_id is not None:
            stats = [
                {"name": s.name, "grad_norm": s.grad_norm, "weight_norm": s.weight_norm, "update_ratio": s.update_ratio}
                for s in layer_stats
            ]
            self._db.layer_stats.log(self._run_id, self._step_count, stats)

    def broadcast_activations(self, stats: list[ActivationStatRecord]):
        """Write per-layer activation stats to DB."""
        if self._db is not None and self._run_id is not None:
            records = [
                {"name": s.name, "mean": s.mean, "std": s.std, "min": s.min, "max": s.max, "pct_zero": s.pct_zero}
                for s in stats
            ]
            self._db.activation_stats.log(self._run_id, self._step_count, records)

    def broadcast_eval(self, step: int, results: dict):
        """Eval results are logged via eval_db, not the main DB. No-op here."""
        pass

    def broadcast_generation(self, step: int, prompt: str, output: str):
        """Write a sample generation to DB."""
        if self._db is not None and self._run_id is not None:
            self._db.generations.log(self._run_id, step, prompt, output)

    def broadcast_stage(
        self,
        stage_index: int,
        stage_name: str,
        total_stages: int,
        dataset: str | None = None,
        stage_type: str = "pretrain",
    ):
        """Write stage info to run_state table."""
        if self._db is not None and self._run_id is not None:
            self._db.run_state.set_stage(
                self._run_id,
                stage_index=stage_index,
                stage_name=stage_name,
                total_stages=total_stages,
                dataset=dataset,
                stage_type=stage_type,
            )
            parts = [f"Stage {stage_index + 1}/{total_stages}: {stage_name}"]
            if dataset:
                parts.append(f"· {dataset}")
            if stage_type != "pretrain":
                parts.append(f"· {stage_type.upper()}")
            self.broadcast_text_state(" ".join(parts))
        self.log(f"=== Stage {stage_index + 1}/{total_stages}: {stage_name} ===")

    def broadcast_text_state(self, text: str):
        """Write text state to run_state table."""
        if self._db is not None and self._run_id is not None:
            self._db.run_state.set_text_state(self._run_id, text)

    def broadcast_status(self, status: WireStatus):
        """Write training status to run_state table."""
        if self._db is not None and self._run_id is not None:
            self._db.run_state.set_status(self._run_id, status)

    def log(self, message: str, level: str = "info"):
        """Write log line to DB and print to console."""
        if self._db is not None:
            self._db.logs.log(self._run_id, level, message)
        print(message, file=sys.stderr)

    def _print_metrics(self, metrics: dict, step: int):
        parts = [f"step={step}"]
        for k, v in metrics.items():
            if k == "train/step":
                continue
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            elif isinstance(v, int):
                parts.append(f"{k}={v}")
        line = " | ".join(parts)
        self.log(line)

    def close(self):
        pass
