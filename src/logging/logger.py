"""Logging: SQLite + console + live broadcast."""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config.base import LoggingConfig
    from src.server.broadcast import Broadcaster
    from src.storage.database import Database
    from src.types.status import WireStatus
    from src.types.ws import LogLevel


class Logger:
    def __init__(
        self,
        config: LoggingConfig,
        experiment_name: str,
        db: Database | None = None,
        run_id: int | None = None,
        broadcaster: Broadcaster | None = None,
    ):
        self.config = config
        self._db = db
        self._run_id = run_id
        self._broadcaster = broadcaster
        self._step_count = 0

    def log_step(self, metrics: dict, step: int | None = None):
        """Log metrics to SQLite, broadcast to WebSocket, and print to console."""
        if step is None:
            step = metrics.get("train/step", self._step_count)
        self._step_count = step

        # SQLite
        if self._db is not None and self._run_id is not None:
            self._db.log_metrics(self._run_id, step, metrics)

        # Live broadcast to dashboard
        if self._broadcaster is not None:
            self._broadcaster.publish({
                "type": "step",
                "data": _to_step_metrics(step, metrics),
            })

        # Console
        if step % self.config.console_interval == 0:
            self._print_metrics(metrics, step)

    def broadcast_layers(self, layer_stats: list[dict]):
        """Broadcast per-layer gradient/weight stats to dashboard."""
        if self._broadcaster is not None:
            self._broadcaster.publish({
                "type": "layers",
                "data": layer_stats,
            })

    def broadcast_generation(self, step: int, prompt: str, output: str):
        """Broadcast a sample generation to dashboard."""
        if self._broadcaster is not None:
            self._broadcaster.publish({
                "type": "generation",
                "data": {"step": step, "prompt": prompt, "output": output},
            })

    def broadcast_status(self, status: WireStatus):
        """Broadcast training status change to dashboard."""
        if self._broadcaster is not None:
            self._broadcaster.publish({
                "type": "status",
                "data": status,
            })

    def log(self, message: str, level: LogLevel = "info"):
        """Broadcast a log line to the dashboard and print to console."""
        if self._broadcaster is not None:
            self._broadcaster.publish({
                "type": "log",
                "data": {"level": level, "message": message},
            })
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


def _to_step_metrics(step: int, m: dict) -> dict:
    """Convert internal metric keys to dashboard StepMetrics shape."""
    step_time = m.get("train/step_time", 0)
    tokens_seen = int(m.get("train/tokens_seen", 0))
    tokens_per_sec = tokens_seen / step_time if step_time > 0 else 0

    result = {
        "step": step,
        "trainLoss": m.get("train/loss", 0),
        "lr": m.get("train/lr", 0),
        "gradNorm": m.get("train/grad_norm", 0),
        "tokensSeen": tokens_seen,
        "tokensPerSec": round(tokens_per_sec),
        "stepTime": round(step_time, 4) if step_time else 0,
    }
    if "val/loss" in m:
        result["valLoss"] = m["val/loss"]
    if "train/loss" in m:
        result["bpc"] = round(m["train/loss"] / math.log(2), 4)
    return result
