"""Evaluation task registry."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.evaluation.tasks.base import EvalTask

_REGISTRY: dict[str, type[EvalTask]] = {}


def register_task(name: str):
    """Decorator to register a task class."""
    def wrapper(cls: type[EvalTask]) -> type[EvalTask]:
        _REGISTRY[name] = cls
        return cls
    return wrapper


def get_task(name: str) -> EvalTask:
    """Instantiate a registered task by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown eval task: {name!r}. Available: {available}")
    return _REGISTRY[name]()


def list_tasks() -> list[str]:
    """Return all registered task names."""
    return sorted(_REGISTRY.keys())


# Import task modules to trigger registration.
from src.evaluation.tasks import perplexity as _perplexity  # noqa: F401, E402
from src.evaluation.tasks import blimp as _blimp  # noqa: F401, E402
from src.evaluation.tasks import state_tracking as _state_tracking  # noqa: F401, E402
from src.evaluation.tasks import generation_quality as _generation_quality  # noqa: F401, E402
from src.evaluation.tasks import quick_loss as _quick_loss  # noqa: F401, E402
from src.evaluation.tasks import ablation_suite as _ablation_suite  # noqa: F401, E402
from src.evaluation.tasks import hellaswag as _hellaswag  # noqa: F401, E402
from src.evaluation.tasks import arc as _arc  # noqa: F401, E402
from src.evaluation.tasks import piqa as _piqa  # noqa: F401, E402
from src.evaluation.tasks import winogrande as _winogrande  # noqa: F401, E402
from src.evaluation.tasks import mmlu as _mmlu  # noqa: F401, E402
from src.evaluation.tasks import lambada as _lambada  # noqa: F401, E402
