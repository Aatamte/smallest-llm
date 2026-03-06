"""Optimizer and scheduler factory with proper weight decay handling."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, LRScheduler

if TYPE_CHECKING:
    from src.config.base import OptimizerConfig, SchedulerConfig


class Muon(Optimizer):
    """Muon optimizer: Momentum + Orthogonalization.

    For each parameter with dim >= 2, applies Newton-Schulz orthogonalization
    to the momentum buffer, giving a better-conditioned update direction.
    1D params (biases, norms) fall back to standard SGD with momentum.

    Converges in 2-5x fewer steps than AdamW on small models.

    Reference: Keller Jordan, "Muon: An optimizer for hidden layers" (2024)
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.95,
                 weight_decay: float = 0.0, ns_steps: int = 5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @staticmethod
    def _newton_schulz(M: torch.Tensor, steps: int = 5) -> torch.Tensor:
        """Approximate orthogonalization via Newton-Schulz iteration.

        Given M, finds the closest orthogonal matrix (in Frobenius norm)
        via iterating: X_{k+1} = X_k (aI + bX_k^T X_k + cX_k^T X_k X_k^T X_k)
        with coefficients chosen for cubic convergence.
        """
        # Coefficients for cubic convergence
        a, b, c = 3.4445, -4.7750, 2.0315

        # Normalize to unit spectral norm (approximately)
        X = M / (M.norm() + 1e-7)
        if X.shape[0] > X.shape[1]:
            X = X.T
            transposed = True
        else:
            transposed = False

        for _ in range(steps):
            A = X @ X.T
            X = a * X + b * (A @ X) + c * (A @ (A @ X))

        if transposed:
            X = X.T
        return X

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            mu = group["momentum"]
            wd = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad

                # Weight decay (decoupled)
                if wd > 0:
                    p.mul_(1 - lr * wd)

                # Get or init momentum buffer
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = g.clone()
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(g)

                if p.dim() >= 2:
                    # Orthogonalize the momentum for matrix params
                    # Reshape 3D+ tensors (conv weights) to 2D for Newton-Schulz
                    orig_shape = buf.shape
                    buf_2d = buf.float().reshape(orig_shape[0], -1)
                    update = self._newton_schulz(buf_2d, steps=ns_steps).reshape(orig_shape).to(buf.dtype)
                    # Scale to preserve the norm ratio
                    update.mul_(max(1, orig_shape[0] / (buf_2d.shape[1])) ** 0.5)
                    p.add_(update, alpha=-lr)
                else:
                    # 1D params: normalized SGD with momentum
                    # (prevents explosion when gradient clipping is disabled)
                    buf_norm = buf.norm()
                    if buf_norm > 1e-7:
                        p.add_(buf / buf_norm, alpha=-lr)
                    else:
                        p.add_(buf, alpha=-lr)

        return loss


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
    elif config.name == "muon":
        return Muon(
            groups,
            lr=config.lr,
            momentum=config.beta1,
        )
    raise ValueError(f"Unknown optimizer: {config.name}. Available: adamw, muon")


def set_optimizer_lr(optimizer: Optimizer, lr: float):
    """Update the base LR in all param groups.

    Sets both 'lr' and 'initial_lr' — the latter is required by LambdaLR
    which multiplies initial_lr by the lambda value to get the actual LR.
    """
    for pg in optimizer.param_groups:
        pg["lr"] = lr
        pg["initial_lr"] = lr


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
