"""Tests for the config system."""

import json
import os
import tempfile

from src.config.base import ExperimentConfig, apply_cli_overrides


def test_defaults():
    config = ExperimentConfig()
    assert config.seed == 42
    assert config.optimizer.lr == 3e-4
    assert config.data.batch_size == 32


def test_to_dict_roundtrip():
    config = ExperimentConfig(name="test_run", seed=123)
    d = config.to_dict()
    restored = ExperimentConfig.from_dict(d)
    assert restored.name == "test_run"
    assert restored.seed == 123
    assert restored.optimizer.lr == config.optimizer.lr
    assert restored.training.max_steps == config.training.max_steps


def test_save_load_roundtrip():
    config = ExperimentConfig(name="save_test")
    config.optimizer.lr = 1e-2
    config.training.max_steps = 500

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name

    try:
        config.save(path)
        loaded = ExperimentConfig.load(path)
        assert loaded.name == "save_test"
        assert loaded.optimizer.lr == 1e-2
        assert loaded.training.max_steps == 500
    finally:
        os.unlink(path)


def test_cli_overrides():
    config = ExperimentConfig()
    overrides = ["--optimizer.lr", "1e-2", "--training.max_steps", "500", "--name", "cli_test"]
    config = apply_cli_overrides(config, overrides)
    assert config.optimizer.lr == 1e-2
    assert config.training.max_steps == 500
    assert config.name == "cli_test"


def test_cli_override_bool():
    config = ExperimentConfig()
    overrides = ["--training.mixed_precision", "true"]
    config = apply_cli_overrides(config, overrides)
    assert config.training.mixed_precision is True
