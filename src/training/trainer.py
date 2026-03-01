"""Core training loop."""

from __future__ import annotations

import time
from itertools import cycle
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader

from src.logging.logger import Logger
from src.training.callbacks import CallbackBase
from src.training.checkpointing import CheckpointManager
from src.utils.device import empty_cache, resolve_device, synchronize

if TYPE_CHECKING:
    from src.config.base import ExperimentConfig


class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        logger: Logger | None = None,
        callbacks: list | None = None,
        db=None,
        run_id: int | None = None,
    ):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.callbacks = callbacks or []

        self.device = resolve_device(config.device)
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint, config, db=db, run_id=run_id,
        )

        self.start_step = 0
        self.tokens_seen = 0
        self.should_stop = False
        self._train_iter = None

    def _next_batch(self) -> dict:
        """Get next batch from infinite iterator."""
        if self._train_iter is None:
            self._train_iter = iter(cycle(self.train_loader))
        return next(self._train_iter)

    def _fire_callback(self, method: str, **kwargs):
        for cb in self.callbacks:
            fn = getattr(cb, method, None)
            if fn:
                fn(self, **kwargs)

    def train(self):
        """Main training loop."""
        tc = self.config.training
        self._fire_callback("on_train_begin")

        for step in range(self.start_step, tc.max_steps):
            if self.should_stop:
                break

            self._fire_callback("on_step_begin", step=step)
            t0 = time.perf_counter()

            # --- Forward / backward with gradient accumulation ---
            self.model.train()
            accumulated_loss = 0.0

            for _ in range(tc.gradient_accumulation_steps):
                batch = self._next_batch()
                batch = {k: v.to(self.device) for k, v in batch.items()}

                output = self.model(**batch)
                loss = output["loss"] if isinstance(output, dict) else output.loss
                loss = loss / tc.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()

                self.tokens_seen += batch["input_ids"].numel()

            # --- Gradient clipping ---
            grad_norm = 0.0
            if self.config.optimizer.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimizer.grad_clip_norm,
                ).item()

            self.optimizer.step()
            self.scheduler.step()

            # --- Callbacks (before zero_grad so gradients are still available) ---
            self._fire_callback("on_step_end", step=step)

            self.optimizer.zero_grad(set_to_none=True)

            dt = time.perf_counter() - t0

            # --- Logging ---
            if step % tc.log_interval == 0 and self.logger:
                self.logger.log_step(
                    {
                        "train/loss": accumulated_loss,
                        "train/grad_norm": grad_norm,
                        "train/lr": self.scheduler.get_last_lr()[0],
                        "train/step": step,
                        "train/tokens_seen": self.tokens_seen,
                        "train/step_time": dt,
                    },
                    step=step,
                )

            # --- Evaluation ---
            if step > 0 and step % tc.eval_interval == 0:
                val_metrics = self.evaluate()
                if self.logger:
                    self.logger.log_step(val_metrics, step=step)
                self._fire_callback("on_eval_end", step=step, metrics=val_metrics)

                # Checkpoint on eval
                if step % tc.save_interval == 0:
                    self.checkpoint_manager.save(
                        step, self.model, self.optimizer, self.scheduler,
                        val_metrics, self.tokens_seen,
                    )

        self._fire_callback("on_train_end")
        if self.logger:
            self.logger.close()

    def evaluate(self) -> dict:
        """Run validation loop."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        max_batches = self.config.data.max_eval_batches

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                output = self.model(**batch)
                loss = output["loss"] if isinstance(output, dict) else output.loss
                total_loss += loss.item()
                num_batches += 1
                if max_batches > 0 and num_batches >= max_batches:
                    break

        self.model.train()
        empty_cache(self.device)

        avg_loss = total_loss / max(num_batches, 1)
        return {"val/loss": avg_loss}

    def resume(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        state = self.checkpoint_manager.load(checkpoint_path, self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.start_step = state["step"] + 1
        self.tokens_seen = state.get("tokens_seen", 0)
        print(f"Resumed from step {state['step']}")
