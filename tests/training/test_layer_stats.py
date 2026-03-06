"""Tests for LayerStatsCallback and the broadcast pipeline."""

import torch
import torch.nn as nn

from src.training.callbacks import LayerStatsCallback


class _FakeLogger:
    """Captures broadcast_layers calls."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    def broadcast_layers(self, stats: list[dict]):
        self.calls.append(stats)


class _FakeTrainer:
    """Minimal trainer stub for testing callbacks."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.logger = _FakeLogger()


def _make_model():
    """Simple 2-layer model."""
    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    return model


def _run_fake_step(model):
    """Run a forward+backward pass to populate gradients."""
    x = torch.randn(2, 4)
    out = model(x)
    loss = out.sum()
    loss.backward()


def test_callback_fires_on_interval():
    model = _make_model()
    _run_fake_step(model)

    cb = LayerStatsCallback(log_interval=10)
    trainer = _FakeTrainer(model)

    # Step 5 — should NOT fire (5 % 10 != 0)
    cb.on_step_end(trainer, step=5)
    assert len(trainer.logger.calls) == 0

    # Step 10 — SHOULD fire
    cb.on_step_end(trainer, step=10)
    assert len(trainer.logger.calls) == 1


def test_callback_returns_correct_fields():
    model = _make_model()
    _run_fake_step(model)

    cb = LayerStatsCallback(log_interval=1)
    trainer = _FakeTrainer(model)
    cb.on_step_end(trainer, step=0)

    stats = trainer.logger.calls[0]
    assert len(stats) > 0

    for record in stats:
        entry = record.to_wire()
        assert "name" in entry
        assert "gradNorm" in entry
        assert "weightNorm" in entry
        assert "updateRatio" in entry
        assert isinstance(entry["gradNorm"], float)
        assert isinstance(entry["weightNorm"], float)
        assert entry["gradNorm"] >= 0
        assert entry["weightNorm"] >= 0


def test_callback_skips_params_without_grad():
    model = _make_model()
    # Freeze first layer — no gradients
    for p in model[0].parameters():
        p.requires_grad = False
    _run_fake_step(model)

    cb = LayerStatsCallback(log_interval=1)
    trainer = _FakeTrainer(model)
    cb.on_step_end(trainer, step=0)

    stats = trainer.logger.calls[0]
    names = [s.to_wire()["name"] for s in stats]
    # Only layer 2 params should appear
    assert all("2" in n for n in names)


def test_update_ratio_nonzero_after_second_call():
    model = _make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    cb = LayerStatsCallback(log_interval=1)
    trainer = _FakeTrainer(model)

    # First step
    _run_fake_step(model)
    cb.on_step_end(trainer, step=0)
    first_stats = trainer.logger.calls[0]
    # First call has no prev weights, so update ratio should be 0
    assert all(s.to_wire()["updateRatio"] == 0.0 for s in first_stats)

    # Actually update weights
    optimizer.step()
    optimizer.zero_grad()

    # Second step
    _run_fake_step(model)
    cb.on_step_end(trainer, step=1)
    second_stats = trainer.logger.calls[1]
    # After optimizer step, weights changed, so update ratio should be > 0
    assert any(s.to_wire()["updateRatio"] > 0 for s in second_stats)


def test_no_logger_does_not_crash():
    model = _make_model()
    _run_fake_step(model)

    cb = LayerStatsCallback(log_interval=1)
    trainer = _FakeTrainer(model)
    trainer.logger = None

    # Should not raise
    cb.on_step_end(trainer, step=0)


def test_end_to_end_with_real_trainer():
    """Verify LayerStatsCallback fires through the real Trainer training loop."""
    from src.config.base import ExperimentConfig
    from src.training.run import build_trainer

    config = ExperimentConfig()
    config.training.max_steps = 3
    config.training.log_interval = 1
    config.training.eval_interval = 9999  # skip evals
    config.training.save_interval = 9999

    trainer, run_id = build_trainer(config)

    # Verify callback is registered
    cb_types = [type(cb).__name__ for cb in trainer.callbacks]
    assert "LayerStatsCallback" in cb_types, f"Expected LayerStatsCallback in {cb_types}"

    trainer.train()

    # Check that layer stats were written to DB
    db = trainer.logger._db
    rows = db.layer_stats.get(run_id)
    assert len(rows) > 0, "No layer stats written to DB"

    # Verify structure
    row = rows[0]
    assert "layer" in row
    assert "grad_norm" in row
    assert "weight_norm" in row
    assert "update_ratio" in row
