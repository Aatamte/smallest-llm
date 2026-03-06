"""Estimate training duration by profiling real steps."""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import cycle

import torch

from src.config.base import ExperimentConfig
from src.data.datasets import HFStreamingDataset, TextFileDataset, load_dataset
from src.data.streaming import build_streaming_dataloaders
from src.data.text import build_dataloaders
from src.data.tokenizer import build_tokenizer
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.run import _build_model
from src.utils.device import resolve_device, synchronize
from src.utils.reproducibility import set_seed


@dataclass
class TrainingEstimate:
    """Results from a training duration estimate."""

    step_time_ms: float
    total_steps: int
    estimated_seconds: float
    param_count: int
    tokens_per_step: int
    tokens_per_second: float

    @property
    def estimated_human(self) -> str:
        """Format estimated_seconds as a human-readable string."""
        s = self.estimated_seconds
        if s < 60:
            return f"{s:.0f}s"
        elif s < 3600:
            m, sec = divmod(s, 60)
            return f"{int(m)}m {int(sec)}s"
        else:
            h, rem = divmod(s, 3600)
            m, sec = divmod(rem, 60)
            return f"{int(h)}h {int(m)}m {int(sec)}s"

    def summary(self) -> str:
        return (
            f"Parameters: {self.param_count:,}\n"
            f"Step time:  {self.step_time_ms:.1f} ms\n"
            f"Total steps: {self.total_steps:,}\n"
            f"Tokens/step: {self.tokens_per_step:,}\n"
            f"Tokens/sec:  {self.tokens_per_second:,.0f}\n"
            f"Estimated:   {self.estimated_human}"
        )


def estimate_training_time(
    config: ExperimentConfig,
    warmup_steps: int = 3,
    timed_steps: int = 10,
) -> TrainingEstimate:
    """Profile a config by running real forward/backward steps, then extrapolate.

    This builds the full pipeline (model, data, optimizer) and runs a small
    number of steps to measure wall-clock time. The estimate includes all real
    overhead: data loading, device sync, gradient accumulation, etc.

    Args:
        config: The experiment config to estimate.
        warmup_steps: Steps to run before timing (device warm-up, JIT, caches).
        timed_steps: Steps to time for the average.

    Returns:
        TrainingEstimate with per-step timing and total duration projection.
    """
    set_seed(config.seed)
    device = resolve_device(config.device)

    # Build data pipeline
    dataset = load_dataset(config.data.dataset_name)
    if isinstance(dataset, TextFileDataset):
        tokenizer = build_tokenizer(config.data.tokenizer_name, text=dataset.text)
        train_loader, _, _ = build_dataloaders(config.data, dataset.text, tokenizer)
    elif isinstance(dataset, HFStreamingDataset):
        tokenizer = build_tokenizer(config.data.tokenizer_name)
        train_loader, _, _ = build_streaming_dataloaders(
            config.data, dataset, tokenizer
        )
    else:
        raise ValueError(f"Unexpected dataset type: {type(dataset)}")

    # Build model + optimizer (the real ones, not mocks)
    model = _build_model(config, tokenizer.vocab_size).to(device)
    optimizer = build_optimizer(config.optimizer, model)
    scheduler = build_scheduler(config.scheduler, optimizer, config.training.max_steps)

    param_count = sum(p.numel() for p in model.parameters())
    tokens_per_step = config.data.batch_size * config.data.max_seq_len * config.training.gradient_accumulation_steps

    # Run steps
    train_iter = iter(cycle(train_loader))
    total_profile_steps = warmup_steps + timed_steps

    def _run_one_step():
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config.training.gradient_accumulation_steps):
            batch = next(train_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            loss = output.loss / config.training.gradient_accumulation_steps
            loss.backward()
        if config.optimizer.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.optimizer.grad_clip_norm
            )
        optimizer.step()
        scheduler.step()

    # Warmup (discard timings — cold-start overhead is not representative)
    for _ in range(warmup_steps):
        _run_one_step()
    synchronize(device)

    # Timed steps
    step_times = []
    for _ in range(timed_steps):
        synchronize(device)
        t0 = time.perf_counter()
        _run_one_step()
        synchronize(device)
        step_times.append(time.perf_counter() - t0)

    avg_step_time = sum(step_times) / len(step_times)
    estimated_total = avg_step_time * config.training.max_steps
    tokens_per_second = tokens_per_step / avg_step_time if avg_step_time > 0 else 0

    return TrainingEstimate(
        step_time_ms=avg_step_time * 1000,
        total_steps=config.training.max_steps,
        estimated_seconds=estimated_total,
        param_count=param_count,
        tokens_per_step=tokens_per_step,
        tokens_per_second=tokens_per_second,
    )
