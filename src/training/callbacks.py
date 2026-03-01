"""Callback protocol and built-in callbacks."""

from __future__ import annotations

import torch
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

        stats = []
        for name, param in trainer.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                weight_norm = param.data.norm(2).item()

                # update ratio = how much weights changed relative to their magnitude
                prev_norm = self._prev_weights.get(name)
                if prev_norm is not None and weight_norm > 0:
                    update_ratio = abs(weight_norm - prev_norm) / weight_norm
                else:
                    update_ratio = 0.0
                self._prev_weights[name] = weight_norm

                # Shorten name: remove common prefixes
                short = name.replace("model.", "").replace(".weight", ".W").replace(".bias", ".b")

                stats.append({
                    "name": short,
                    "gradNorm": round(grad_norm, 6),
                    "weightNorm": round(weight_norm, 4),
                    "updateRatio": round(update_ratio, 8),
                })

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
            cls._BLOCK_TYPES = (TransformerBlock, MambaBlock)
        return cls._BLOCK_TYPES

    def _make_hook(self, name: str):
        def hook_fn(module, input, output):
            # output may be a tensor or tuple; grab the tensor
            t = output if isinstance(output, torch.Tensor) else output[0]
            t = t.detach().float()
            numel = t.numel()
            self._stats[name] = {
                "name": name,
                "mean": round(t.mean().item(), 6),
                "std": round(t.std().item(), 6),
                "max": round(t.max().item(), 4),
                "min": round(t.min().item(), 4),
                "pctZero": round((t == 0).sum().item() / max(numel, 1) * 100, 2),
            }
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
        # Send stats sorted by module name
        stats = sorted(self._stats.values(), key=lambda s: s["name"])
        trainer.logger.broadcast_activations(stats)
        self._stats.clear()

    def on_train_end(self, trainer: Trainer) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._stats.clear()
