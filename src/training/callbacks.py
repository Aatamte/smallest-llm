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


class EvalCallback(CallbackBase):
    """Run eval tasks periodically during training and log results as metrics.

    Hooks into on_eval_end (fires after val loss). Checks if the current step
    aligns with eval_interval before running (so eval tasks run less frequently
    than val loss).
    """

    def __init__(
        self,
        tasks: list[str],
        eval_interval: int = 2000,
        max_samples: int | None = 2000,
        data_dir: str = "data/eval",
        eval_db=None,
        run_id: int | None = None,
        model_name: str | None = None,
    ):
        self.tasks = tasks
        self.eval_interval = eval_interval
        self.max_samples = max_samples
        self.data_dir = data_dir
        self.eval_db = eval_db
        self.run_id = run_id
        self.model_name = model_name

    def on_eval_end(self, trainer: Trainer, step: int, metrics: dict) -> None:
        if step == 0 or step % self.eval_interval != 0:
            return

        from src.evaluation.config import EvalConfig
        from src.evaluation.runner import evaluate
        from src.evaluation.trainer_model import TrainerModelWrapper

        total_tasks = len(self.tasks)
        tasks_str = ", ".join(self.tasks)
        prefix = trainer._stage_prefix()
        if trainer.logger:
            trainer.logger.broadcast_text_state(
                f"{prefix}Running evals · Step {step:,} · {tasks_str}"
            )

        model_wrapper = TrainerModelWrapper(
            trainer.model, trainer.tokenizer, trainer.device,
        )

        # Run each task individually so we can update text state per-task
        all_results: dict = {}
        for i, task_name in enumerate(self.tasks):
            if trainer.logger:
                trainer.logger.broadcast_text_state(
                    f"{prefix}Running eval {i + 1}/{total_tasks}: {task_name} · Step {step:,}"
                )

            single_config = EvalConfig(
                tasks=[task_name],
                max_samples=self.max_samples,
                data_dir=self.data_dir,
            )

            def _progress(task_index, task_count, task_name, current, total):
                if trainer.logger:
                    trainer.logger.broadcast_text_state(
                        f"{prefix}Eval {task_index + 1}/{task_count}: {task_name} · "
                        f"Sample {current}/{total} · Step {step:,}"
                    )

            task_results = evaluate(
                model_wrapper,
                single_config,
                db=self.eval_db,
                run_id=self.run_id,
                step=step,
                model_name=self.model_name or trainer.config.name,
                on_progress=_progress,
            )
            all_results.update(task_results)

        # Flatten results into metrics dict for logging
        eval_metrics: dict[str, float] = {}
        eval_broadcast: dict[str, dict] = {}
        for task_name, result in all_results.items():
            for metric_key, value in result.metrics.items():
                eval_metrics[f"eval/{task_name}/{metric_key}"] = value
            eval_broadcast[task_name] = result.metrics

        if eval_metrics and trainer.logger:
            trainer.logger.log_step(eval_metrics, step=step)
            trainer.logger.broadcast_eval(step, eval_broadcast)
