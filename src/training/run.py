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
from src.models import build_model
from src.storage import Database
import math

from src.training.callbacks import ActivationStatsCallback, EvalCallback, LayerStatsCallback
from src.training.techniques.echo_loss import EchoHeads
from src.training.techniques.grad_sharpen import sharpen_gradients
from src.training.techniques.hydra import HydraCallback
from src.training.techniques.multi_token import MultiTokenHeads
from src.training.techniques.neuroplasticity import GrowthStage, NeuroplasticityCallback
from src.training.techniques.phantom import PhantomConfig
from src.training.techniques.state_anchor import StateAnchorConfig, StateAnchorHeads
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.trainer import Trainer
from src.utils.device import resolve_device
from src.utils.env import get_env_info
from src.utils.reproducibility import set_seed

if TYPE_CHECKING:
    from src.server.broadcast import Broadcaster
    from src.training.pipeline import PipelineRunner


def _build_shared_components(
    config: ExperimentConfig,
    db: Database | None = None,
    broadcaster: Broadcaster | None = None,
    run_id: int | None = None,
) -> dict:
    # Enable TF32 for faster matmuls on Ampere+ GPUs (no-op on other hardware)
    torch.set_float32_matmul_precision("high")
    """Build components shared between single-stage and pipeline runs.

    Returns a dict with: config, device, dataset, tokenizer, model, optimizer,
    db, run_id, logger, callbacks.
    """
    set_seed(config.seed)
    device = resolve_device(config.device)

    # Data
    dataset = load_dataset(config.data.dataset_name)
    if isinstance(dataset, TextFileDataset):
        tokenizer = build_tokenizer(config.data.tokenizer_name, text=dataset.text)
    elif isinstance(dataset, HFStreamingDataset):
        tokenizer = build_tokenizer(config.data.tokenizer_name)
    else:
        raise ValueError(f"Unexpected dataset type: {type(dataset)}")

    # Model — start small if neuroplasticity is enabled
    if config.training.neuroplasticity:
        from copy import deepcopy
        small_config = deepcopy(config)
        args = small_config.model.extra_args
        target_d = args.get("d_model", 128)
        target_layers = args.get("n_layers", 7)
        # Start at ~1/4 size
        args["d_model"] = max(16, target_d // 4)
        args["n_layers"] = 1
        d_state = args.get("d_state", 16)
        args["d_state"] = max(4, d_state // 4)
        # Ensure even d_state for RoPE
        args["d_state"] = args["d_state"] + (args["d_state"] % 2)
        model = _build_model(small_config, tokenizer.vocab_size).to(device)
        print(f"Model (neuroplasticity init): {model.count_parameters():,} params on {device}")
        print(f"    Starting at d_model={args['d_model']}, n_layers=1, d_state={args['d_state']}")
        print(f"    Target: d_model={target_d}, n_layers={target_layers}, d_state={d_state}")
    else:
        model = _build_model(config, tokenizer.vocab_size).to(device)
        print(f"Model: {model.count_parameters():,} params on {device}")

    # torch.compile (if supported and requested)
    if config.training.compile_model:
        try:
            model = torch.compile(model)
            print("Model compiled with torch.compile")
        except Exception as e:
            print(f"torch.compile not available, skipping: {e}")

    # Echo heads (auxiliary prediction heads for echo loss)
    # Use the INITIAL model's d_model (may be small if neuroplasticity)
    echo_heads = None
    if config.training.echo_loss:
        initial_d_model = model.d_model if hasattr(model, "d_model") else config.model.extra_args.get("d_model", 64)
        echo_heads = EchoHeads(initial_d_model, tokenizer.vocab_size).to(device)
        echo_params = sum(p.numel() for p in echo_heads.parameters())
        print(f"Echo heads: {echo_params:,} params (backward + skip-2,4,8)")

    # State anchor heads
    anchor_heads = None
    if config.training.state_anchor:
        distances = tuple(int(x) for x in config.training.state_anchor_distances.split(","))
        anchor_config = StateAnchorConfig(
            distances=distances,
            weight=config.training.state_anchor_weight,
        )
        initial_d_model = model.d_model if hasattr(model, "d_model") else config.model.extra_args.get("d_model", 64)
        anchor_heads = StateAnchorHeads(initial_d_model, anchor_config).to(device)
        anchor_params = sum(p.numel() for p in anchor_heads.parameters())
        print(f"State anchor heads: {anchor_params:,} params (distances={distances})")

    # Multi-token prediction heads
    multi_token_heads = None
    if config.training.multi_token:
        initial_d_model = model.d_model if hasattr(model, "d_model") else config.model.extra_args.get("d_model", 64)
        multi_token_heads = MultiTokenHeads(
            initial_d_model, tokenizer.vocab_size,
            n_ahead=config.training.multi_token_n_ahead,
        ).to(device)
        mt_params = sum(p.numel() for p in multi_token_heads.parameters())
        print(f"Multi-token heads: {mt_params:,} params (n_ahead={config.training.multi_token_n_ahead})")

    # Optimizer — include echo heads and anchor heads params if present
    optimizer = build_optimizer(config.optimizer, model)
    if echo_heads is not None:
        optimizer.add_param_group({
            "params": list(echo_heads.parameters()),
            "weight_decay": 0.0,
        })
    if anchor_heads is not None:
        optimizer.add_param_group({
            "params": list(anchor_heads.parameters()),
            "weight_decay": 0.0,
        })
    if multi_token_heads is not None:
        optimizer.add_param_group({
            "params": list(multi_token_heads.parameters()),
            "weight_decay": 0.0,
        })

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

    # Eval tasks callback
    if config.training.eval_tasks:
        from src.storage import EvalDatabase
        eval_db = EvalDatabase("eval.db")
        task_list = [t.strip() for t in config.training.eval_tasks_list.split(",") if t.strip()]
        max_samples = config.training.eval_tasks_max_samples or None
        eval_cb = EvalCallback(
            tasks=task_list,
            eval_interval=config.training.eval_tasks_interval,
            max_samples=max_samples,
            eval_db=eval_db,
            run_id=run_id,
            model_name=config.name,
        )
        callbacks.append(eval_cb)
        print(f"Eval tasks: {', '.join(task_list)} every {config.training.eval_tasks_interval} steps"
              f" (max_samples={max_samples})")

    # Phantom batches
    phantom_config = None
    if config.training.phantom_batches:
        phantom_config = PhantomConfig(
            n_phantoms=config.training.phantom_n,
            weight=config.training.phantom_weight,
        )
        print(f"Phantom batches: {phantom_config.n_phantoms} phantoms, weight={phantom_config.weight}")

    # Gradient sharpening config
    grad_sharpen_keep = config.training.grad_sharpen_keep if config.training.grad_sharpen else None

    return {
        "config": config,
        "device": device,
        "dataset": dataset,
        "tokenizer": tokenizer,
        "model": model,
        "optimizer": optimizer,
        "echo_heads": echo_heads,
        "phantom_config": phantom_config,
        "anchor_heads": anchor_heads,
        "multi_token_heads": multi_token_heads,
        "grad_sharpen_keep": grad_sharpen_keep,
        "db": db,
        "run_id": run_id,
        "logger": logger,
        "callbacks": callbacks,
    }


def _build_hydra_loaders(config, dataset, tokenizer):
    """Build micro and macro DataLoaders for Hydra training."""
    from copy import deepcopy
    from dataclasses import replace

    tc = config.training
    micro_data = replace(config.data, max_seq_len=tc.hydra_micro_seq_len, batch_size=tc.hydra_micro_batch)
    macro_data = replace(config.data, max_seq_len=tc.hydra_macro_seq_len, batch_size=tc.hydra_macro_batch)

    if isinstance(dataset, TextFileDataset):
        micro_loader, _, _ = build_dataloaders(micro_data, dataset.text, tokenizer)
        macro_loader, _, _ = build_dataloaders(macro_data, dataset.text, tokenizer)
    elif isinstance(dataset, HFStreamingDataset):
        micro_loader, _, _ = build_streaming_dataloaders(micro_data, dataset, tokenizer)
        macro_loader, _, _ = build_streaming_dataloaders(macro_data, dataset, tokenizer)
    else:
        raise ValueError(f"Unexpected dataset type for Hydra: {type(dataset)}")

    return micro_loader, macro_loader


def _build_neuro_schedule(config) -> list[GrowthStage]:
    """Auto-compute neuroplasticity growth schedule from model config.

    Strategy: start at 1/4 target size, grow at 25% and 55% of training.
    """
    args = config.model.extra_args
    target_d = args.get("d_model", 128)
    target_layers = args.get("n_layers", 7)
    target_d_state = args.get("d_state", 16)
    expand = args.get("expand_factor", 2)
    mlp_factor = args.get("mlp_factor", 4)
    max_steps = config.training.max_steps

    stages = []

    # Stage 1 → 2: grow from 1/4 to 1/2 at 25% of steps
    mid_d = max(16, target_d // 2)
    mid_d_state = max(4, target_d_state // 2)
    # Ensure even d_state for RoPE
    mid_d_state = mid_d_state + (mid_d_state % 2)
    mid_layers = max(1, math.ceil(target_layers / 2))
    stages.append(GrowthStage(
        step=int(max_steps * 0.25),
        d_model=mid_d,
        n_layers=mid_layers,
        d_state=mid_d_state,
        expand_factor=expand,
        mlp_factor=mlp_factor,
    ))

    # Stage 2 → 3: grow to full size at 55% of steps
    stages.append(GrowthStage(
        step=int(max_steps * 0.55),
        d_model=target_d,
        n_layers=target_layers,
        d_state=target_d_state,
        expand_factor=expand,
        mlp_factor=mlp_factor,
    ))

    return stages


def _attach_hydra_and_neuro(config, callbacks, dataset, tokenizer):
    """Attach Hydra and Neuroplasticity callbacks if enabled in config."""
    tc = config.training

    if tc.hydra:
        micro_loader, macro_loader = _build_hydra_loaders(config, dataset, tokenizer)
        hydra_cb = HydraCallback(
            micro_loader=micro_loader,
            macro_loader=macro_loader,
            micro_weight=tc.hydra_micro_weight,
            macro_weight=tc.hydra_macro_weight,
            weight_schedule="shift",
            total_steps=tc.max_steps,
        )
        callbacks.append(hydra_cb)
        print(f"Hydra: micro seq={tc.hydra_micro_seq_len} batch={tc.hydra_micro_batch}, "
              f"macro seq={tc.hydra_macro_seq_len} batch={tc.hydra_macro_batch}")

    if tc.neuroplasticity:
        schedule = _build_neuro_schedule(config)
        neuro_cb = NeuroplasticityCallback(schedule)
        callbacks.append(neuro_cb)
        for s in schedule:
            print(f"Neuro: step {s.step} → d_model={s.d_model}, n_layers={s.n_layers}, d_state={s.d_state}")


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

    c = _build_shared_components(config, db, broadcaster, run_id)

    # Build dataloaders for single-stage run
    dataset = c["dataset"]
    tokenizer = c["tokenizer"]
    if isinstance(dataset, TextFileDataset):
        train_loader, val_loader, _ = build_dataloaders(
            config.data, dataset.text, tokenizer
        )
    elif isinstance(dataset, HFStreamingDataset):
        train_loader, val_loader, _ = build_streaming_dataloaders(
            config.data, dataset, tokenizer
        )
    else:
        raise ValueError(f"Unexpected dataset type: {type(dataset)}")

    scheduler = build_scheduler(config.scheduler, c["optimizer"], config.training.max_steps)

    # Attach Hydra + Neuroplasticity callbacks if enabled
    _attach_hydra_and_neuro(config, c["callbacks"], dataset, tokenizer)

    trainer = Trainer(
        config=config,
        model=c["model"],
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=c["optimizer"],
        scheduler=scheduler,
        logger=c["logger"],
        callbacks=c["callbacks"],
        checkpoint_db=checkpoint_db,
        run_id=c["run_id"],
        tokenizer=tokenizer,
        echo_heads=c["echo_heads"],
        phantom_config=c["phantom_config"],
        anchor_heads=c["anchor_heads"],
        multi_token_heads=c["multi_token_heads"],
        grad_sharpen_keep=c["grad_sharpen_keep"],
    )

    return trainer, c["run_id"]


def build_pipeline_runner(
    config: ExperimentConfig,
    db: Database | None = None,
    broadcaster: Broadcaster | None = None,
    run_id: int | None = None,
    checkpoint_db=None,
) -> tuple[PipelineRunner, Trainer, int]:
    """Build a PipelineRunner for multi-stage training.

    Returns (pipeline_runner, trainer, run_id). The trainer is the underlying
    Trainer instance (useful for stop signals and model export).
    """
    from src.training.pipeline import PipelineRunner

    c = _build_shared_components(config, db, broadcaster, run_id)

    dataset = c["dataset"]
    tokenizer = c["tokenizer"]

    # Build initial dataloaders (will be rebuilt per-stage by PipelineRunner)
    if isinstance(dataset, TextFileDataset):
        train_loader, val_loader, _ = build_dataloaders(
            config.data, dataset.text, tokenizer
        )
    elif isinstance(dataset, HFStreamingDataset):
        train_loader, val_loader, _ = build_streaming_dataloaders(
            config.data, dataset, tokenizer
        )
    else:
        raise ValueError(f"Unexpected dataset type: {type(dataset)}")

    scheduler = build_scheduler(config.scheduler, c["optimizer"], config.training.max_steps)

    # Attach Hydra + Neuroplasticity callbacks if enabled
    _attach_hydra_and_neuro(config, c["callbacks"], dataset, tokenizer)

    trainer = Trainer(
        config=config,
        model=c["model"],
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=c["optimizer"],
        scheduler=scheduler,
        logger=c["logger"],
        callbacks=c["callbacks"],
        checkpoint_db=checkpoint_db,
        run_id=c["run_id"],
        tokenizer=tokenizer,
        echo_heads=c["echo_heads"],
        phantom_config=c["phantom_config"],
        anchor_heads=c["anchor_heads"],
        multi_token_heads=c["multi_token_heads"],
        grad_sharpen_keep=c["grad_sharpen_keep"],
    )

    runner = PipelineRunner(config, trainer, dataset, tokenizer)
    return runner, trainer, c["run_id"]


def start_run(
    config: Optional[ExperimentConfig] = None,
    db: Optional[Database] = None,
    broadcaster: Optional[Broadcaster] = None,
    run_id: Optional[int] = None,
) -> int:
    """Build and run training. Returns the run_id.

    Automatically dispatches to pipeline runner if config.stages is set.
    """
    if config is None:
        config = ExperimentConfig()

    if config.stages:
        runner, trainer, run_id = build_pipeline_runner(config, db, broadcaster, run_id)
        try:
            runner.run()
            db = db or Database(config.logging.db_path)
            db.finish_run(run_id, status="completed")
            if broadcaster:
                trainer.logger.broadcast_status("complete")
        except Exception:
            db = db or Database(config.logging.db_path)
            db.finish_run(run_id, status="failed")
            if broadcaster:
                trainer.logger.broadcast_status("idle")
            raise
        return run_id

    trainer, run_id = build_trainer(config, db, broadcaster, run_id)
    try:
        trainer.train()
        db = db or Database(config.logging.db_path)
        db.finish_run(run_id, status="completed")
        if broadcaster:
            trainer.logger.broadcast_status("complete")
    except Exception:
        db = db or Database(config.logging.db_path)
        db.finish_run(run_id, status="failed")
        if broadcaster:
            trainer.logger.broadcast_status("idle")
        raise

    return run_id


def _build_model(config: ExperimentConfig, vocab_size: int) -> torch.nn.Module:
    """Build a model from config via the model registry."""
    return build_model(
        config.model.name,
        vocab_size=vocab_size,
        max_seq_len=config.data.max_seq_len,
        extra_args=config.model.extra_args,
    )
