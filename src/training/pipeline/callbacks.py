"""Pipeline stage callbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.training.callbacks import CallbackBase

if TYPE_CHECKING:
    from src.training.trainer import Trainer


class SanityCheckCallback(CallbackBase):
    """Stops training when loss drops below a threshold."""

    def __init__(self, loss_threshold: float):
        self.loss_threshold = loss_threshold
        self._last_loss: float | None = None

    def on_step_end(self, trainer: Trainer, step: int) -> None:
        pass

    def on_eval_end(self, trainer: Trainer, step: int, metrics: dict) -> None:
        val_loss = metrics.get("val/loss")
        if val_loss is not None and val_loss < self.loss_threshold:
            print(
                f"Sanity check PASSED at step {step}: "
                f"val/loss={val_loss:.4f} < {self.loss_threshold}"
            )
            trainer.should_stop = True


class StageMetadataCallback(CallbackBase):
    """Injects stage info onto the trainer so metrics include stage context."""

    def __init__(self, stage_index: int, stage_name: str, total_stages: int, dataset_name: str):
        self.stage_index = stage_index
        self.stage_name = stage_name
        self.total_stages = total_stages
        self.dataset_name = dataset_name

    def on_train_begin(self, trainer: Trainer) -> None:
        trainer._current_stage_index = self.stage_index
        trainer._current_stage_name = self.stage_name
        trainer._total_stages = self.total_stages
        trainer._current_dataset = self.dataset_name

    def on_train_end(self, trainer: Trainer) -> None:
        for attr in ("_current_stage_index", "_current_stage_name", "_total_stages", "_current_dataset"):
            if hasattr(trainer, attr):
                delattr(trainer, attr)
