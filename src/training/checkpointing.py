"""Checkpoint save/load/rotate with RNG state preservation."""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Optional

import torch

from src.utils.reproducibility import get_rng_states, set_rng_states

if TYPE_CHECKING:
    from src.config.base import CheckpointConfig, ExperimentConfig


class CheckpointManager:
    def __init__(
        self,
        config: CheckpointConfig,
        experiment_config: ExperimentConfig,
        db=None,
        run_id: int | None = None,
    ):
        self.config = config
        self.experiment_config = experiment_config
        self.save_dir = os.path.join(config.save_dir, experiment_config.name)
        os.makedirs(self.save_dir, exist_ok=True)
        self.best_value = float("inf") if config.best_mode == "min" else float("-inf")
        self._db = db
        self._run_id = run_id

    def save(
        self,
        step: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        metrics: dict,
        tokens_seen: int,
        flops_total: int = 0,
    ):
        """Save checkpoint with full state for exact resumption."""
        state = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": self.experiment_config.to_dict(),
            "metrics": metrics,
            "tokens_seen": tokens_seen,
            "flops_total": flops_total,
            "rng_states": get_rng_states(),
        }

        path = os.path.join(self.save_dir, f"checkpoint-{step}.pt")
        torch.save(state, path)

        # Update best
        is_best = False
        if self.config.save_best and self.config.best_metric in metrics:
            val = metrics[self.config.best_metric]
            is_better = (
                val < self.best_value
                if self.config.best_mode == "min"
                else val > self.best_value
            )
            if is_better:
                self.best_value = val
                is_best = True
                best_path = os.path.join(self.save_dir, "best.pt")
                torch.save(state, best_path)

        # Log to database
        if self._db is not None and self._run_id is not None:
            self._db.log_checkpoint(
                self._run_id, step, path, metrics, is_best=is_best,
            )

        # Rotate old checkpoints
        self._rotate()

    def load(self, path: str, device: torch.device) -> dict:
        """Load checkpoint and restore RNG states."""
        state = torch.load(path, map_location=device, weights_only=False)
        if "rng_states" in state:
            set_rng_states(state["rng_states"])
        return state

    def find_latest(self) -> Optional[str]:
        """Find the most recent checkpoint."""
        if not os.path.exists(self.save_dir):
            return None
        checkpoints = self._list_checkpoints()
        return checkpoints[-1] if checkpoints else None

    def _list_checkpoints(self) -> list[str]:
        """List checkpoint files sorted by step number."""
        pattern = re.compile(r"checkpoint-(\d+)\.pt$")
        entries = []
        for fname in os.listdir(self.save_dir):
            m = pattern.match(fname)
            if m:
                entries.append((int(m.group(1)), os.path.join(self.save_dir, fname)))
        entries.sort(key=lambda x: x[0])
        return [path for _, path in entries]

    def _rotate(self):
        """Delete old checkpoints beyond keep_last_n."""
        checkpoints = self._list_checkpoints()
        while len(checkpoints) > self.config.keep_last_n:
            os.remove(checkpoints.pop(0))
