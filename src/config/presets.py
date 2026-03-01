"""Named preset configurations for quick experiment setup."""

from __future__ import annotations

from src.config.base import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)

PRESETS: dict[str, ExperimentConfig] = {
    "default": ExperimentConfig(
        name="default",
    ),
    "transformer-quick": ExperimentConfig(
        name="transformer-quick",
        model=ModelConfig(name="transformer", extra_args={"d_model": 64, "n_heads": 2, "n_layers": 2}),
        data=DataConfig(max_seq_len=64, batch_size=32),
        training=TrainingConfig(max_steps=200, eval_interval=50, log_interval=5, save_interval=100),
        optimizer=OptimizerConfig(lr=1e-3),
        scheduler=SchedulerConfig(warmup_steps=20),
    ),
    "transformer-long": ExperimentConfig(
        name="transformer-long",
        model=ModelConfig(name="transformer", extra_args={"d_model": 128, "n_heads": 4, "n_layers": 4}),
        data=DataConfig(max_seq_len=256, batch_size=32),
        training=TrainingConfig(max_steps=5000, eval_interval=250, log_interval=10, save_interval=500),
        optimizer=OptimizerConfig(lr=3e-4),
        scheduler=SchedulerConfig(warmup_steps=100),
    ),
    "mamba-default": ExperimentConfig(
        name="mamba-default",
        model=ModelConfig(name="mamba", extra_args={"d_model": 128, "n_layers": 7, "d_state": 16, "d_conv": 4, "expand_factor": 2}),
        data=DataConfig(max_seq_len=128, batch_size=32),
        training=TrainingConfig(max_steps=1000, eval_interval=100, log_interval=5, save_interval=200),
        optimizer=OptimizerConfig(lr=3e-4),
        scheduler=SchedulerConfig(warmup_steps=50),
    ),
    "mamba-quick": ExperimentConfig(
        name="mamba-quick",
        model=ModelConfig(name="mamba", extra_args={"d_model": 64, "n_layers": 4, "d_state": 16, "d_conv": 4, "expand_factor": 2}),
        data=DataConfig(max_seq_len=64, batch_size=32),
        training=TrainingConfig(max_steps=200, eval_interval=50, log_interval=5, save_interval=100),
        optimizer=OptimizerConfig(lr=1e-3),
        scheduler=SchedulerConfig(warmup_steps=20),
    ),
}

# Human-readable labels for the frontend
PRESET_LABELS: dict[str, str] = {
    "default": "Default (Transformer)",
    "transformer-quick": "Transformer Quick (200 steps)",
    "transformer-long": "Transformer Long (5K steps)",
    "mamba-default": "Mamba Default (1K steps)",
    "mamba-quick": "Mamba Quick (200 steps)",
}


def get_presets() -> list[dict[str, str]]:
    """Return list of {name, label} for all presets."""
    return [{"name": k, "label": PRESET_LABELS.get(k, k)} for k in PRESETS]


def get_preset(name: str) -> ExperimentConfig | None:
    """Return a preset config by name, or None if not found."""
    return PRESETS.get(name)
