"""Tests for trainer, optimizer, scheduler, checkpointing."""

import os
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.base import ExperimentConfig
from src.data.text import build_dataloaders
from src.data.tokenizer import build_tokenizer
from src.models.base import BaseModel, ModelOutput
from src.training.callbacks import EarlyStoppingCallback
from src.training.checkpointing import CheckpointManager
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.trainer import Trainer


class TinyModel(BaseModel):
    def __init__(self, vocab_size, dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embed(input_ids)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return ModelOutput(loss=loss, logits=logits)


def _make_components(max_steps=10, device="cpu"):
    config = ExperimentConfig(name="test", device=device)
    config.training.max_steps = max_steps
    config.training.log_interval = 5
    config.training.eval_interval = 5
    config.training.save_interval = 5
    config.data.batch_size = 2
    config.data.max_seq_len = 8
    config.logging.use_tensorboard = False

    text = "the quick brown fox jumps over the lazy dog " * 50
    tok = build_tokenizer("char", text)
    train_loader, val_loader, _ = build_dataloaders(config.data, text, tok)
    model = TinyModel(tok.vocab_size).to(device)
    optimizer = build_optimizer(config.optimizer, model)
    scheduler = build_scheduler(config.scheduler, optimizer, max_steps)
    return config, model, train_loader, val_loader, optimizer, scheduler


def test_optimizer_param_groups():
    """Weight decay should not apply to biases and low-dim params."""
    config = ExperimentConfig()
    model = TinyModel(10)
    optimizer = build_optimizer(config.optimizer, model)
    # Two param groups: decay and no_decay
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == config.optimizer.weight_decay
    assert optimizer.param_groups[1]["weight_decay"] == 0.0


def test_scheduler_warmup():
    config = ExperimentConfig()
    model = TinyModel(10)
    optimizer = build_optimizer(config.optimizer, model)
    scheduler = build_scheduler(config.scheduler, optimizer, total_steps=1000)
    # LR should increase during warmup
    lrs = []
    for _ in range(config.scheduler.warmup_steps):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()
    assert lrs[-1] > lrs[0]


def test_trainer_runs():
    config, model, train_loader, val_loader, optimizer, scheduler = _make_components(
        max_steps=10
    )
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    trainer.train()
    assert trainer.tokens_seen > 0


def test_early_stopping():
    cb = EarlyStoppingCallback(patience=2, metric="val/loss")
    # Simulate no improvement
    class FakeTrainer:
        should_stop = False
    t = FakeTrainer()
    cb.on_eval_end(t, step=1, metrics={"val/loss": 1.0})
    assert not t.should_stop
    cb.on_eval_end(t, step=2, metrics={"val/loss": 1.0})
    assert not t.should_stop
    cb.on_eval_end(t, step=3, metrics={"val/loss": 1.0})
    assert t.should_stop


def test_checkpoint_save_load():
    config, model, train_loader, val_loader, optimizer, scheduler = _make_components()

    with tempfile.TemporaryDirectory() as tmpdir:
        config.checkpoint.save_dir = tmpdir
        mgr = CheckpointManager(config.checkpoint, config)

        # Save
        mgr.save(5, model, optimizer, scheduler, {"val/loss": 0.5}, tokens_seen=100)
        assert mgr.find_latest() is not None

        # Load
        state = mgr.load(mgr.find_latest(), torch.device("cpu"))
        assert state["step"] == 5
        assert state["tokens_seen"] == 100
        assert "model_state_dict" in state
        assert "optimizer_state_dict" in state


def test_checkpoint_rotation():
    config, model, train_loader, val_loader, optimizer, scheduler = _make_components()

    with tempfile.TemporaryDirectory() as tmpdir:
        config.checkpoint.save_dir = tmpdir
        config.checkpoint.keep_last_n = 2
        mgr = CheckpointManager(config.checkpoint, config)

        for step in [1, 2, 3, 4]:
            mgr.save(step, model, optimizer, scheduler, {"val/loss": 0.5}, tokens_seen=0)

        checkpoints = mgr._list_checkpoints()
        assert len(checkpoints) == 2
        assert "checkpoint-3" in checkpoints[0]
        assert "checkpoint-4" in checkpoints[1]
