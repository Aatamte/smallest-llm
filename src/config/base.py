"""Configuration dataclasses for the training infrastructure."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from src.evaluation.config import EvalConfig


@dataclass
class ModelConfig:
    """Placeholder — each architecture defines its own params via extra_args."""
    name: str = "transformer"
    extra_args: dict = field(default_factory=dict)


@dataclass
class DataConfig:
    dataset_name: str = "tiny_shakespeare"
    tokenizer_name: str = "char"
    max_seq_len: int = 128
    batch_size: int = 32
    num_workers: int = 0  # 0 safest on MPS
    train_split: float = 0.9
    val_split: float = 0.05
    test_split: float = 0.05
    max_eval_batches: int = 0  # 0 = all batches; set >0 to cap eval for streaming


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip_norm: float = 1.0


@dataclass
class SchedulerConfig:
    name: str = "cosine_with_warmup"
    warmup_steps: int = 50
    min_lr_ratio: float = 0.1


@dataclass
class TrainingConfig:
    max_steps: int = 1_000
    eval_interval: int = 100
    log_interval: int = 5
    save_interval: int = 200
    gradient_accumulation_steps: int = 1
    mixed_precision: bool = False
    compile_model: bool = False


@dataclass
class CheckpointConfig:
    save_dir: str = "checkpoints"
    keep_last_n: int = 3
    save_best: bool = True
    best_metric: str = "val/loss"
    best_mode: str = "min"


@dataclass
class LoggingConfig:
    db_path: str = "smallest_llm.db"
    console_interval: int = 10


@dataclass
class ExperimentConfig:
    """Top-level config that composes everything."""
    name: str = "default"
    seed: int = 42
    device: str = "auto"

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> ExperimentConfig:
        """Reconstruct from a dict, handling nested dataclasses."""
        nested_types = {
            "model": ModelConfig,
            "data": DataConfig,
            "optimizer": OptimizerConfig,
            "scheduler": SchedulerConfig,
            "training": TrainingConfig,
            "checkpoint": CheckpointConfig,
            "logging": LoggingConfig,
            "eval": EvalConfig,
        }
        kwargs = {}
        for f in fields(cls):
            if f.name in d:
                if f.name in nested_types:
                    kwargs[f.name] = nested_types[f.name](**d[f.name])
                else:
                    kwargs[f.name] = d[f.name]
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str) -> ExperimentConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))


def apply_cli_overrides(config: ExperimentConfig, overrides: list[str]) -> ExperimentConfig:
    """Apply dot-notation CLI overrides like --optimizer.lr 1e-3."""
    d = config.to_dict()
    i = 0
    while i < len(overrides):
        key = overrides[i].lstrip("-")
        if i + 1 >= len(overrides):
            break
        value = overrides[i + 1]
        i += 2

        # Navigate dot notation
        parts = key.split(".")
        target = d
        for part in parts[:-1]:
            target = target[part]

        # Auto-cast the value
        target[parts[-1]] = _auto_cast(value, target.get(parts[-1]))

    return ExperimentConfig.from_dict(d)


def _auto_cast(value: str, existing: Any) -> Any:
    """Cast string value to match the type of the existing value."""
    if existing is None:
        return value
    if isinstance(existing, bool):
        return value.lower() in ("true", "1", "yes")
    if isinstance(existing, int):
        return int(value)
    if isinstance(existing, float):
        return float(value)
    return value
