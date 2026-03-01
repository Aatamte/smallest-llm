"""Evaluation package for smallest-llm."""

from src.evaluation.config import EvalConfig
from src.evaluation.protocol import Evaluatable
from src.evaluation.results import EvalResult
from src.evaluation.runner import evaluate
from src.evaluation.tasks import get_task, list_tasks

__all__ = [
    "EvalConfig",
    "EvalResult",
    "Evaluatable",
    "evaluate",
    "get_task",
    "list_tasks",
]
