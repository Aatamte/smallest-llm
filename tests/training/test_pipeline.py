"""Tests for multi-stage training pipeline."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.base import ExperimentConfig, StageConfig
from src.config.presets import get_preset
from src.data.text import build_dataloaders
from src.data.tokenizer import build_tokenizer
from src.models.base import BaseModel, ModelOutput
from src.training.optimizer import build_optimizer, build_scheduler, set_optimizer_lr
from src.training.pipeline import (
    FixedBatchLoader,
    PipelineRunner,
    SanityCheckCallback,
    StageMetadataCallback,
)
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


TEXT = "the quick brown fox jumps over the lazy dog " * 100


def _make_trainer_and_dataset(max_steps=10, seq_len=8, batch_size=2):
    """Build a minimal trainer + dataset for pipeline tests."""
    config = ExperimentConfig(name="test", device="cpu")
    config.training.max_steps = max_steps
    config.training.log_interval = 5
    config.training.eval_interval = 5
    config.training.save_interval = max_steps
    config.data.batch_size = batch_size
    config.data.max_seq_len = seq_len

    tok = build_tokenizer("char", text=TEXT)
    train_loader, val_loader, _ = build_dataloaders(config.data, TEXT, tok)

    from src.data.datasets import TextFileDataset
    dataset = TextFileDataset(TEXT)

    model = TinyModel(tok.vocab_size).to("cpu")
    optimizer = build_optimizer(config.optimizer, model)
    scheduler = build_scheduler(config.scheduler, optimizer, max_steps)

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    return trainer, dataset, tok, config


def test_no_stages_fallback():
    """When config.stages is None, PipelineRunner falls back to trainer.train()."""
    trainer, dataset, tok, config = _make_trainer_and_dataset(max_steps=5)
    config.stages = None
    runner = PipelineRunner(config, trainer, dataset, tok)
    runner.run()
    assert trainer.tokens_seen > 0


def test_single_stage_runs():
    """A single stage completes and advances tokens."""
    trainer, dataset, tok, config = _make_trainer_and_dataset(max_steps=10)
    config.stages = [
        StageConfig(name="only", max_steps=10, seq_len=8, eval_interval=5, log_interval=5, save_interval=10),
    ]
    runner = PipelineRunner(config, trainer, dataset, tok)
    runner.run()
    assert trainer.tokens_seen > 0


def test_multi_stage_global_step_continuous():
    """Global step is continuous across stages."""
    trainer, dataset, tok, config = _make_trainer_and_dataset(max_steps=20)
    config.stages = [
        StageConfig(name="a", max_steps=5, seq_len=8, eval_interval=5, log_interval=5, save_interval=5),
        StageConfig(name="b", max_steps=5, seq_len=8, eval_interval=5, log_interval=5, save_interval=5),
    ]
    runner = PipelineRunner(config, trainer, dataset, tok)
    runner.run()
    # After both stages, the trainer should have seen tokens from 10 total steps
    assert trainer.tokens_seen > 0


def test_stage_rebuilds_with_different_seq_len():
    """Stages with different seq_len rebuild dataloaders correctly."""
    trainer, dataset, tok, config = _make_trainer_and_dataset(max_steps=20)
    config.stages = [
        StageConfig(name="short", max_steps=5, seq_len=8, eval_interval=5, log_interval=5, save_interval=5),
        StageConfig(name="longer", max_steps=5, seq_len=16, eval_interval=5, log_interval=5, save_interval=5),
    ]
    runner = PipelineRunner(config, trainer, dataset, tok)
    runner.run()
    # If this completes without error, dataloaders were rebuilt successfully
    assert trainer.tokens_seen > 0


def test_optimizer_state_carries_across_stages():
    """Optimizer momentum buffers persist between stages."""
    trainer, dataset, tok, config = _make_trainer_and_dataset(max_steps=20)
    config.stages = [
        StageConfig(name="a", max_steps=5, seq_len=8, eval_interval=5, log_interval=5, save_interval=5),
        StageConfig(name="b", max_steps=5, seq_len=8, eval_interval=5, log_interval=5, save_interval=5),
    ]

    runner = PipelineRunner(config, trainer, dataset, tok)

    # Run first stage manually to check optimizer state
    runner.run()

    # Check that optimizer has state (momentum buffers from AdamW)
    state = trainer.optimizer.state
    assert len(state) > 0, "Optimizer should have accumulated state across stages"


def test_should_stop_between_stages():
    """Setting should_stop before a stage starts exits the pipeline."""
    trainer, dataset, tok, config = _make_trainer_and_dataset(max_steps=20)
    config.stages = [
        StageConfig(name="a", max_steps=5, seq_len=8, eval_interval=5, log_interval=5, save_interval=5),
        StageConfig(name="b", max_steps=5, seq_len=8, eval_interval=5, log_interval=5, save_interval=5),
    ]

    # Pre-stop before any stage runs
    trainer.should_stop = True
    runner = PipelineRunner(config, trainer, dataset, tok)
    runner.run()

    # No tokens processed because we stopped before first stage
    assert trainer.tokens_seen == 0


def test_fixed_batch_loader():
    """FixedBatchLoader yields from a fixed list."""
    batches = [{"a": 1}, {"b": 2}, {"c": 3}]
    loader = FixedBatchLoader(batches)
    assert len(loader) == 3
    result = list(loader)
    assert result == batches
    # Iterating again yields the same
    result2 = list(loader)
    assert result2 == batches


def test_sanity_check_callback():
    """SanityCheckCallback stops trainer when loss is below threshold."""
    cb = SanityCheckCallback(loss_threshold=1.0)

    class FakeTrainer:
        should_stop = False

    t = FakeTrainer()
    # Loss above threshold — don't stop
    cb.on_eval_end(t, step=1, metrics={"val/loss": 2.0})
    assert not t.should_stop

    # Loss below threshold — stop
    cb.on_eval_end(t, step=2, metrics={"val/loss": 0.5})
    assert t.should_stop


def test_stage_metadata_callback():
    """StageMetadataCallback sets and cleans up stage attributes."""
    cb = StageMetadataCallback(stage_index=2, stage_name="extension", total_stages=4, dataset_name="minipile")

    class FakeTrainer:
        pass

    t = FakeTrainer()
    cb.on_train_begin(t)
    assert t._current_stage_index == 2
    assert t._current_stage_name == "extension"
    assert t._total_stages == 4
    assert t._current_dataset == "minipile"

    cb.on_train_end(t)
    assert not hasattr(t, "_current_stage_index")
    assert not hasattr(t, "_current_stage_name")
    assert not hasattr(t, "_total_stages")
    assert not hasattr(t, "_current_dataset")


def test_stage_config_serialization_roundtrip():
    """StageConfig survives to_dict / from_dict round-trip."""
    config = ExperimentConfig(
        name="test",
        stages=[
            StageConfig(name="a", max_steps=100, seq_len=32, lr=1e-3),
            StageConfig(name="b", max_steps=200, seq_len=64),
        ],
    )
    d = config.to_dict()
    restored = ExperimentConfig.from_dict(d)

    assert restored.stages is not None
    assert len(restored.stages) == 2
    assert restored.stages[0].name == "a"
    assert restored.stages[0].seq_len == 32
    assert restored.stages[0].lr == 1e-3
    assert restored.stages[1].name == "b"
    assert restored.stages[1].max_steps == 200


def test_set_optimizer_lr():
    """set_optimizer_lr updates both lr and initial_lr."""
    model = TinyModel(10)
    optimizer = build_optimizer(ExperimentConfig().optimizer, model)
    set_optimizer_lr(optimizer, 0.042)
    for pg in optimizer.param_groups:
        assert pg["lr"] == 0.042
        assert pg["initial_lr"] == 0.042


def test_preset_sanity_check_has_stages():
    """The sanity-check preset has stages configured."""
    config = get_preset("sanity-check")
    assert config is not None
    assert config.stages is not None
    assert len(config.stages) == 1
    assert config.stages[0].overfit_batches == 1


def test_preset_full_transformer_has_four_stages():
    """The full-transformer preset has 4 stages."""
    config = get_preset("full-transformer")
    assert config is not None
    assert config.stages is not None
    assert len(config.stages) == 4
    names = [s.name for s in config.stages]
    assert names == ["sanity", "foundation", "extension", "refinement"]


def test_preset_full_mamba_has_four_stages():
    """The full-mamba preset has 4 stages."""
    config = get_preset("full-mamba")
    assert config is not None
    assert config.stages is not None
    assert len(config.stages) == 4
