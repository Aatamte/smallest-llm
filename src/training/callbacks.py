"""Callback protocol and built-in callbacks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.training.trainer import Trainer


@runtime_checkable
class Callback(Protocol):
    def on_train_begin(self, trainer: Trainer) -> None: ...
    def on_train_end(self, trainer: Trainer) -> None: ...
    def on_step_begin(self, trainer: Trainer, step: int) -> None: ...
    def on_step_end(self, trainer: Trainer, step: int) -> None: ...
    def on_eval_end(self, trainer: Trainer, step: int, metrics: dict) -> None: ...


class CallbackBase:
    """No-op base. Override what you need."""

    def on_train_begin(self, trainer: Trainer) -> None:
        pass

    def on_train_end(self, trainer: Trainer) -> None:
        pass

    def on_step_begin(self, trainer: Trainer, step: int) -> None:
        pass

    def on_step_end(self, trainer: Trainer, step: int) -> None:
        pass

    def on_eval_end(self, trainer: Trainer, step: int, metrics: dict) -> None:
        pass


class EarlyStoppingCallback(CallbackBase):
    """Stop training if val loss hasn't improved in `patience` evals."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4, metric: str = "val/loss"):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.best_value = float("inf")
        self.wait = 0

    def on_eval_end(self, trainer: Trainer, step: int, metrics: dict) -> None:
        value = metrics.get(self.metric)
        if value is None:
            return
        if value < self.best_value - self.min_delta:
            self.best_value = value
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"Early stopping at step {step} (no improvement for {self.patience} evals)")
                trainer.should_stop = True


class GradientStatsCallback(CallbackBase):
    """Log per-layer gradient norms. Invaluable for debugging tiny models."""

    def __init__(self, log_interval: int = 100):
        self.log_interval = log_interval

    def on_step_end(self, trainer: Trainer, step: int) -> None:
        if step % self.log_interval != 0:
            return
        grad_stats = {}
        for name, param in trainer.model.named_parameters():
            if param.grad is not None:
                grad_stats[f"grad_norm/{name}"] = param.grad.data.norm(2).item()
        if trainer.logger:
            trainer.logger.log_step(grad_stats, step=step)
