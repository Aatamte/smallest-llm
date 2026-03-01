"""Assemble and launch a training run from config."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from src.config.base import ExperimentConfig
from src.data.datasets import HFStreamingDataset, TextFileDataset, load_dataset
from src.data.streaming import build_streaming_dataloaders
from src.data.text import build_dataloaders
from src.data.tokenizer import build_tokenizer
from src.logging.logger import Logger
from src.models.mamba import TinyMamba
from src.models.tiny_transformer import TinyTransformer
from src.storage.database import Database
from src.training.callbacks import ActivationStatsCallback, LayerStatsCallback
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.utils.device import resolve_device
from src.utils.env import get_env_info
from src.utils.reproducibility import set_seed

if TYPE_CHECKING:
    from src.server.broadcast import Broadcaster


def build_trainer(
    config: Optional[ExperimentConfig] = None,
    db: Optional[Database] = None,
    broadcaster: Optional[Broadcaster] = None,
    run_id: Optional[int] = None,
    checkpoint_db=None,
) -> tuple[Trainer, int]:
    """Build everything from config. Returns (trainer, run_id) without starting."""
    if config is None:
        config = ExperimentConfig()

    set_seed(config.seed)
    device = resolve_device(config.device)

    # Data
    dataset = load_dataset(config.data.dataset_name)
    if isinstance(dataset, TextFileDataset):
        tokenizer = build_tokenizer(config.data.tokenizer_name, text=dataset.text)
        train_loader, val_loader, _ = build_dataloaders(
            config.data, dataset.text, tokenizer
        )
    elif isinstance(dataset, HFStreamingDataset):
        tokenizer = build_tokenizer(config.data.tokenizer_name)
        train_loader, val_loader, _ = build_streaming_dataloaders(
            config.data, dataset, tokenizer
        )
    else:
        raise ValueError(f"Unexpected dataset type: {type(dataset)}")

    # Model
    model = _build_model(config, tokenizer.vocab_size).to(device)

    print(f"Model: {model.count_parameters():,} params on {device}")

    # Optimizer + scheduler
    optimizer = build_optimizer(config.optimizer, model)
    scheduler = build_scheduler(config.scheduler, optimizer, config.training.max_steps)

    # Database + logging
    if db is None:
        db = Database(config.logging.db_path)

    if run_id is None:
        env_info = get_env_info()
        run_id = db.create_run(config.name, config.to_dict(), env_info)

    logger = Logger(
        config.logging,
        config.name,
        db=db,
        run_id=run_id,
        broadcaster=broadcaster,
    )

    if broadcaster:
        logger.broadcast_status("training")

    callbacks = [LayerStatsCallback(log_interval=config.training.log_interval)]
    if config.monitoring.activation_stats:
        callbacks.append(ActivationStatsCallback(log_interval=config.monitoring.log_interval))

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        callbacks=callbacks,
        checkpoint_db=checkpoint_db,
        run_id=run_id,
    )

    return trainer, run_id


def start_run(
    config: Optional[ExperimentConfig] = None,
    db: Optional[Database] = None,
    broadcaster: Optional[Broadcaster] = None,
    run_id: Optional[int] = None,
) -> int:
    """Build and run training. Returns the run_id."""
    trainer, run_id = build_trainer(config, db, broadcaster, run_id)

    try:
        trainer.train()
        db.finish_run(run_id, status="completed")
        if broadcaster:
            trainer.logger.broadcast_status("complete")
    except Exception as e:
        db.finish_run(run_id, status="failed")
        if broadcaster:
            trainer.logger.broadcast_status("idle")
        raise

    return run_id


def _build_model(config: ExperimentConfig, vocab_size: int) -> torch.nn.Module:
    """Build a model from config."""
    args = config.model.extra_args
    name = config.model.name

    if name == "transformer":
        return TinyTransformer(
            vocab_size=vocab_size,
            d_model=args.get("d_model", 64),
            n_heads=args.get("n_heads", 2),
            n_layers=args.get("n_layers", 2),
            max_seq_len=config.data.max_seq_len,
            dropout=args.get("dropout", 0.1),
        )
    elif name == "mamba":
        return TinyMamba(
            vocab_size=vocab_size,
            d_model=args.get("d_model", 128),
            n_layers=args.get("n_layers", 7),
            d_state=args.get("d_state", 16),
            d_conv=args.get("d_conv", 4),
            expand_factor=args.get("expand_factor", 2),
            dt_rank=args.get("dt_rank", 0),
            max_seq_len=config.data.max_seq_len,
            dropout=args.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown model: {name!r}. Available: transformer, mamba")
