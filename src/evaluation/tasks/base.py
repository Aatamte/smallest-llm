"""Base class for evaluation tasks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional

from src.evaluation.config import EvalConfig
from src.evaluation.protocol import Evaluatable
from src.evaluation.results import EvalResult

# on_progress(current_sample, total_samples)
ProgressCallback = Callable[[int, int], None]


class EvalTask(ABC):
    """Abstract base for all evaluation tasks."""

    name: str

    @abstractmethod
    def download(self, data_dir: str) -> None:
        """Download / cache the dataset if not already present."""

    @abstractmethod
    def evaluate(
        self,
        model: Evaluatable,
        config: EvalConfig,
        on_progress: Optional[ProgressCallback] = None,
    ) -> EvalResult:
        """Run evaluation and return results."""
