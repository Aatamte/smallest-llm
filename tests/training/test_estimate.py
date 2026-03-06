"""Tests for training duration estimation."""

import pytest

from src.config.base import ExperimentConfig, ModelConfig, DataConfig, TrainingConfig, OptimizerConfig, SchedulerConfig
from src.training.estimate import TrainingEstimate, estimate_training_time


def _quick_config(model_name="transformer", max_steps=100):
    """Build a minimal config for fast estimation."""
    extra = {"d_model": 32, "n_heads": 2, "n_layers": 1}
    if model_name == "mamba":
        extra = {"d_model": 32, "n_layers": 1, "d_state": 4, "d_conv": 2, "expand_factor": 2}

    return ExperimentConfig(
        name=f"estimate-test-{model_name}",
        device="cpu",
        model=ModelConfig(name=model_name, extra_args=extra),
        data=DataConfig(
            dataset_name="tiny_shakespeare",
            tokenizer_name="char",
            max_seq_len=16,
            batch_size=4,
        ),
        optimizer=OptimizerConfig(lr=1e-3),
        scheduler=SchedulerConfig(warmup_steps=5),
        training=TrainingConfig(max_steps=max_steps),
    )


class TestTrainingEstimate:
    def test_estimate_transformer(self):
        config = _quick_config("transformer", max_steps=50)
        est = estimate_training_time(config, warmup_steps=1, timed_steps=3)

        assert est.param_count > 0
        assert est.step_time_ms > 0
        assert est.total_steps == 50
        assert est.estimated_seconds > 0
        assert est.tokens_per_step == 4 * 16  # batch_size * seq_len
        assert est.tokens_per_second > 0

    def test_estimate_mamba(self):
        config = _quick_config("mamba", max_steps=50)
        est = estimate_training_time(config, warmup_steps=1, timed_steps=3)

        assert est.param_count > 0
        assert est.step_time_ms > 0
        assert est.estimated_seconds > 0

    def test_estimate_scales_with_steps(self):
        """Estimated time should scale linearly with max_steps."""
        config_short = _quick_config("transformer", max_steps=100)
        config_long = _quick_config("transformer", max_steps=200)

        est_short = estimate_training_time(config_short, warmup_steps=1, timed_steps=3)
        est_long = estimate_training_time(config_long, warmup_steps=1, timed_steps=3)

        # Step times should be similar (same model), total should ~double
        ratio = est_long.estimated_seconds / est_short.estimated_seconds
        assert 1.2 < ratio < 4.0  # loose bound — CPU timing has variance

    def test_estimate_with_gradient_accumulation(self):
        """Gradient accumulation should increase tokens_per_step."""
        config = _quick_config("transformer", max_steps=50)
        config.training.gradient_accumulation_steps = 4

        est = estimate_training_time(config, warmup_steps=1, timed_steps=3)

        assert est.tokens_per_step == 4 * 16 * 4  # batch * seq * accum
        assert est.step_time_ms > 0

    def test_human_readable_formatting(self):
        short = TrainingEstimate(
            step_time_ms=10, total_steps=100,
            estimated_seconds=45, param_count=1000,
            tokens_per_step=64, tokens_per_second=6400,
        )
        assert short.estimated_human == "45s"

        medium = TrainingEstimate(
            step_time_ms=10, total_steps=1000,
            estimated_seconds=150, param_count=1000,
            tokens_per_step=64, tokens_per_second=6400,
        )
        assert medium.estimated_human == "2m 30s"

        long = TrainingEstimate(
            step_time_ms=10, total_steps=100000,
            estimated_seconds=7384, param_count=1000,
            tokens_per_step=64, tokens_per_second=6400,
        )
        assert long.estimated_human == "2h 3m 4s"

    def test_summary_string(self):
        est = TrainingEstimate(
            step_time_ms=12.5, total_steps=1000,
            estimated_seconds=12.5, param_count=50000,
            tokens_per_step=512, tokens_per_second=40960,
        )
        s = est.summary()
        assert "50,000" in s
        assert "12.5 ms" in s
        assert "1,000" in s
        assert "512" in s
