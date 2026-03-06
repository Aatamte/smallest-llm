"""Tests for ActivationStatsCallback."""

import torch
import torch.nn as nn

from src.training.callbacks import ActivationStatsCallback


class _FakeLogger:
    def __init__(self):
        self.activation_calls: list[list[dict]] = []
        self.layer_calls: list[list[dict]] = []

    def broadcast_activations(self, stats):
        self.activation_calls.append(stats)

    def broadcast_layers(self, stats):
        self.layer_calls.append(stats)


class _FakeTrainer:
    def __init__(self, model):
        self.model = model
        self.logger = _FakeLogger()


def _make_model():
    """Simple model with a block-like module for hooking."""
    # We need to use actual block types. Use a plain Sequential for unit tests
    # and real models for integration tests.
    return nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )


def _run_forward(model):
    x = torch.randn(2, 4)
    return model(x)


def test_hooks_registered_on_train_begin():
    """Hooks should be registered for block-type modules."""
    from src.models.tiny_transformer import TinyTransformer

    model = TinyTransformer(vocab_size=50, d_model=32, n_heads=2, n_layers=2, max_seq_len=16)
    cb = ActivationStatsCallback(log_interval=1)
    trainer = _FakeTrainer(model)

    assert len(cb._hooks) == 0
    cb.on_train_begin(trainer)
    # Should have hooks for each TransformerBlock (2 blocks)
    assert len(cb._hooks) == 2


def test_hooks_removed_on_train_end():
    from src.models.tiny_transformer import TinyTransformer

    model = TinyTransformer(vocab_size=50, d_model=32, n_heads=2, n_layers=2, max_seq_len=16)
    cb = ActivationStatsCallback(log_interval=1)
    trainer = _FakeTrainer(model)

    cb.on_train_begin(trainer)
    assert len(cb._hooks) == 2

    cb.on_train_end(trainer)
    assert len(cb._hooks) == 0
    assert len(cb._stats) == 0


def test_activation_stats_collected_after_forward():
    from src.models.tiny_transformer import TinyTransformer

    model = TinyTransformer(vocab_size=50, d_model=32, n_heads=2, n_layers=2, max_seq_len=16)
    cb = ActivationStatsCallback(log_interval=1)
    trainer = _FakeTrainer(model)

    cb.on_train_begin(trainer)

    # Run a forward pass to trigger hooks
    x = torch.randint(0, 50, (2, 8))
    model(input_ids=x)

    # Stats should be populated
    assert len(cb._stats) == 2  # 2 blocks


def test_stats_have_correct_fields():
    from src.models.tiny_transformer import TinyTransformer

    model = TinyTransformer(vocab_size=50, d_model=32, n_heads=2, n_layers=2, max_seq_len=16)
    cb = ActivationStatsCallback(log_interval=1)
    trainer = _FakeTrainer(model)

    cb.on_train_begin(trainer)
    x = torch.randint(0, 50, (2, 8))
    model(input_ids=x)

    cb.on_step_end(trainer, step=0)

    assert len(trainer.logger.activation_calls) == 1
    stats = trainer.logger.activation_calls[0]
    assert len(stats) == 2

    for record in stats:
        entry = record.to_wire()
        assert "name" in entry
        assert "mean" in entry
        assert "std" in entry
        assert "max" in entry
        assert "min" in entry
        assert "pctZero" in entry
        assert isinstance(entry["mean"], float)
        assert isinstance(entry["std"], float)
        assert entry["std"] >= 0
        assert entry["pctZero"] >= 0


def test_stats_cleared_after_broadcast():
    from src.models.tiny_transformer import TinyTransformer

    model = TinyTransformer(vocab_size=50, d_model=32, n_heads=2, n_layers=2, max_seq_len=16)
    cb = ActivationStatsCallback(log_interval=1)
    trainer = _FakeTrainer(model)

    cb.on_train_begin(trainer)
    x = torch.randint(0, 50, (2, 8))
    model(input_ids=x)
    cb.on_step_end(trainer, step=0)

    assert len(cb._stats) == 0  # cleared after broadcast


def test_respects_log_interval():
    from src.models.tiny_transformer import TinyTransformer

    model = TinyTransformer(vocab_size=50, d_model=32, n_heads=2, n_layers=2, max_seq_len=16)
    cb = ActivationStatsCallback(log_interval=10)
    trainer = _FakeTrainer(model)

    cb.on_train_begin(trainer)
    x = torch.randint(0, 50, (2, 8))
    model(input_ids=x)

    # Step 5 — should NOT broadcast
    cb.on_step_end(trainer, step=5)
    assert len(trainer.logger.activation_calls) == 0

    # Step 10 — SHOULD broadcast
    cb.on_step_end(trainer, step=10)
    assert len(trainer.logger.activation_calls) == 1


def test_mamba_model_hooks():
    from src.models.mamba import TinyMamba

    model = TinyMamba(vocab_size=50, d_model=32, n_layers=3, max_seq_len=16)
    cb = ActivationStatsCallback(log_interval=1)
    trainer = _FakeTrainer(model)

    cb.on_train_begin(trainer)
    assert len(cb._hooks) == 3  # 3 MambaBlocks

    x = torch.randint(0, 50, (2, 8))
    model(input_ids=x)
    cb.on_step_end(trainer, step=0)

    assert len(trainer.logger.activation_calls) == 1
    assert len(trainer.logger.activation_calls[0]) == 3


def test_no_logger_does_not_crash():
    from src.models.tiny_transformer import TinyTransformer

    model = TinyTransformer(vocab_size=50, d_model=32, n_heads=2, n_layers=2, max_seq_len=16)
    cb = ActivationStatsCallback(log_interval=1)
    trainer = _FakeTrainer(model)
    trainer.logger = None

    cb.on_train_begin(trainer)
    x = torch.randint(0, 50, (2, 8))
    model(input_ids=x)
    cb.on_step_end(trainer, step=0)  # should not raise


def test_disabled_config_skips_callback():
    """When monitoring.activation_stats is False, callback should not be registered."""
    from src.config.base import ExperimentConfig

    config = ExperimentConfig()
    config.monitoring.activation_stats = False
    config.training.max_steps = 2
    config.training.log_interval = 1

    from src.training.run import build_trainer
    trainer, _ = build_trainer(config)

    cb_types = [type(cb).__name__ for cb in trainer.callbacks]
    assert "ActivationStatsCallback" not in cb_types


def test_enabled_config_registers_callback():
    """When monitoring.activation_stats is True, callback should be registered."""
    from src.config.base import ExperimentConfig

    config = ExperimentConfig()
    config.monitoring.activation_stats = True
    config.training.max_steps = 2

    from src.training.run import build_trainer
    trainer, _ = build_trainer(config)

    cb_types = [type(cb).__name__ for cb in trainer.callbacks]
    assert "ActivationStatsCallback" in cb_types


def test_end_to_end_with_real_trainer():
    """Full integration: build_trainer + train + verify activations broadcast."""

    class _CaptureBroadcaster:
        def __init__(self):
            self.messages = []

        def publish(self, msg):
            self.messages.append(msg)

    broadcaster = _CaptureBroadcaster()

    from src.config.base import ExperimentConfig
    config = ExperimentConfig()
    config.training.max_steps = 3
    config.training.log_interval = 1
    config.training.eval_interval = 9999
    config.training.save_interval = 9999
    config.monitoring.activation_stats = True
    config.monitoring.log_interval = 1

    from src.training.run import build_trainer
    trainer, _ = build_trainer(config, broadcaster=broadcaster)
    trainer.train()

    activation_msgs = [m for m in broadcaster.messages if m.get("type") == "activations"]
    assert len(activation_msgs) > 0, (
        f"No activation messages. Types: {[m.get('type') for m in broadcaster.messages]}"
    )

    stats = activation_msgs[0]["data"]
    assert len(stats) > 0
    assert "name" in stats[0]
    assert "mean" in stats[0]
    assert "pctZero" in stats[0]
