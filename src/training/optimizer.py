"""Optimizer and scheduler factory with proper weight decay handling."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

if TYPE_CHECKING:
    from src.config.base import OptimizerConfig, SchedulerConfig


def build_optimizer(config: OptimizerConfig, model: torch.nn.Module) -> Optimizer:
    """Build optimizer with proper weight decay separation."""
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim < 2 or "bias" in name or "norm" in name or "ln" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    if config.name == "adamw":
        return AdamW(
            groups,
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
    raise ValueError(f"Unknown optimizer: {config.name}")


def build_scheduler(
    config: SchedulerConfig, optimizer: Optimizer, total_steps: int
) -> LRScheduler:
    """Cosine annealing with linear warmup."""

    def lr_lambda(step: int) -> float:
        # Linear warmup
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        # Cosine decay
        progress = (step - config.warmup_steps) / max(
            1, total_steps - config.warmup_steps
        )
        return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return LambdaLR(optimizer, lr_lambda)
