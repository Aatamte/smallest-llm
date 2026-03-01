"""Evaluation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """Configuration for running evaluations."""

    tasks: list[str] = field(default_factory=lambda: ["perplexity"])
    batch_size: int = 32
    max_samples: int | None = None  # None = use full dataset
    num_few_shot: int = 0
    seed: int = 42
    data_dir: str = "data/eval"  # cache directory for downloaded datasets
