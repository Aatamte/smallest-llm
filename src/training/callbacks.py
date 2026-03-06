"""Callback protocol and built-in callbacks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from src.types.activation_stat import ActivationStatRecord
from src.types.layer_stat import LayerStatRecord

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


class LayerStatsCallback(CallbackBase):
    """Broadcast per-layer grad/weight stats to the dashboard."""

    def __init__(self, log_interval: int = 50):
        self.log_interval = log_interval
        self._prev_weights: dict[str, float] = {}

    def on_step_end(self, trainer: Trainer, step: int) -> None:
        if step % self.log_interval != 0:
            return
        if not trainer.logger:
            return

        stats: list[LayerStatRecord] = []
        for name, param in trainer.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                weight_norm = param.data.norm(2).item()

                prev_norm = self._prev_weights.get(name)
                if prev_norm is not None and weight_norm > 0:
                    update_ratio = abs(weight_norm - prev_norm) / weight_norm
                else:
                    update_ratio = 0.0
                self._prev_weights[name] = weight_norm

                short = name.replace("model.", "").replace(".weight", ".W").replace(".bias", ".b")

                stats.append(LayerStatRecord(
                    name=short,
                    grad_norm=round(grad_norm, 6),
                    weight_norm=round(weight_norm, 4),
                    update_ratio=round(update_ratio, 8),
                ))

        trainer.logger.broadcast_layers(stats)


class ActivationStatsCallback(CallbackBase):
    """Collect and broadcast per-layer activation statistics via forward hooks."""

    # Block classes we want to hook — imported lazily to avoid circular imports
    _BLOCK_TYPES: tuple[type, ...] | None = None

    def __init__(self, log_interval: int = 50):
        self.log_interval = log_interval
        self._stats: dict[str, dict] = {}
        self._hooks: list[torch.utils.hooks.RemovableHook] = []

    @classmethod
    def _get_block_types(cls) -> tuple[type, ...]:
        if cls._BLOCK_TYPES is None:
            from src.models.tiny_transformer import TransformerBlock
            from src.models.mamba import MambaBlock
            from src.models.mamba2 import Mamba2Block
            from src.models.mamba3 import Mamba3Layer
            from src.models.improved_mamba3 import ImprovedMamba3Layer
            from src.models.plastic_mamba3 import PlasticMamba3Layer
            from src.models.multiscale_mamba3 import MultiScaleMamba3Layer
            from src.models.hybrid_mamba3 import LocalAttentionLayer
            cls._BLOCK_TYPES = (TransformerBlock, MambaBlock, Mamba2Block, Mamba3Layer, ImprovedMamba3Layer, PlasticMamba3Layer, MultiScaleMamba3Layer, LocalAttentionLayer)
        return cls._BLOCK_TYPES

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            t = output if isinstance(output, torch.Tensor) else output[0]
            t = t.detach().float()
            numel = t.numel()
            self._stats[name] = ActivationStatRecord(
                name=name,
                mean=round(t.mean().item(), 6),
                std=round(t.std().item(), 6),
                max=round(t.max().item(), 4),
                min=round(t.min().item(), 4),
                pct_zero=round((t == 0).sum().item() / max(numel, 1) * 100, 2),
            )
        return hook_fn

    def on_train_begin(self, trainer: Trainer) -> None:
        block_types = self._get_block_types()
        for name, module in trainer.model.named_modules():
            if isinstance(module, block_types):
                short = name.replace("model.", "")
                handle = module.register_forward_hook(self._make_hook(short))
                self._hooks.append(handle)

    def on_step_end(self, trainer: Trainer, step: int) -> None:
        if step % self.log_interval != 0:
            return
        if not trainer.logger or not self._stats:
            return
        stats = sorted(self._stats.values(), key=lambda s: s.name)
        trainer.logger.broadcast_activations(stats)
        self._stats.clear()

    def on_train_end(self, trainer: Trainer) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._stats.clear()


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
