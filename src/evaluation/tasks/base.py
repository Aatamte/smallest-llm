"""Base class for evaluation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.evaluation.config import EvalConfig
from src.evaluation.protocol import Evaluatable
from src.evaluation.results import EvalResult


class EvalTask(ABC):
    """Abstract base for all evaluation tasks."""

    name: str

    @abstractmethod
    def download(self, data_dir: str) -> None:
        """Download / cache the dataset if not already present."""

    @abstractmethod
    def evaluate(self, model: Evaluatable, config: EvalConfig) -> EvalResult:
        """Run evaluation and return results."""
