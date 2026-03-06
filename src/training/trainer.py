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
from src.training.techniques.echo_loss import EchoHeads, compute_echo_loss
from src.training.techniques.grad_sharpen import sharpen_gradients
from src.training.techniques.multi_token import MultiTokenHeads, compute_multi_token_loss
from src.training.techniques.phantom import PhantomConfig, compute_phantom_loss
from src.training.techniques.state_anchor import StateAnchorHeads, compute_state_anchor_loss
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
        checkpoint_db=None,
        run_id: int | None = None,
        tokenizer=None,
        echo_heads: EchoHeads | None = None,
        phantom_config: PhantomConfig | None = None,
        anchor_heads: StateAnchorHeads | None = None,
        multi_token_heads: MultiTokenHeads | None = None,
        grad_sharpen_keep: float | None = None,
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.callbacks = callbacks or []
        self.echo_heads = echo_heads
        self.phantom_config = phantom_config
        self.anchor_heads = anchor_heads
        self.multi_token_heads = multi_token_heads
        self.grad_sharpen_keep = grad_sharpen_keep

        # Parse multi-token weights from config string
        self._multi_token_weights = None
        if multi_token_heads is not None:
            w_str = getattr(config.training, "multi_token_weights", "1.0,0.5,0.25,0.125")
            self._multi_token_weights = [float(x) for x in w_str.split(",")]

        self.device = resolve_device(config.device)
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint, config, db=checkpoint_db, run_id=run_id,
        )

        self.start_step = 0
        self.tokens_seen = 0
        self.flops_total = 0
        self.should_stop = False
        self._flops_per_token = 0
        self._max_flops = config.training.max_flops
        self._train_iter = None
        self._cached_val_batches = None
        self._timers: dict[str, float] = {}

        # Mixed precision setup
        self._use_amp = config.training.mixed_precision
        if self._use_amp:
            if self.device.type == "cuda":
                self._amp_dtype = torch.float16
                self._scaler = torch.amp.GradScaler(enabled=True)
            elif self.device.type == "mps":
                self._amp_dtype = torch.float16
                self._scaler = torch.amp.GradScaler(enabled=False)  # MPS doesn't need scaler
            else:
                self._use_amp = False
                self._scaler = None
        else:
            self._scaler = None

    def _tick(self) -> float:
        return time.perf_counter()

    def _tock(self, name: str, t0: float):
        self._timers[name] = self._timers.get(name, 0.0) + (time.perf_counter() - t0)

    def _log_timing_summary(self):
        total = sum(self._timers.values())
        if total == 0:
            return
        print("\n" + "=" * 50)
        print("TIMING BREAKDOWN")
        print("=" * 50)
        for name, secs in sorted(self._timers.items(), key=lambda x: -x[1]):
            pct = secs / total * 100
            print(f"  {name:<12s} {secs:>8.1f}s  ({pct:>5.1f}%)")
        print(f"  {'TOTAL':<12s} {total:>8.1f}s")
        print("=" * 50)

        if self.logger:
            step = self.config.training.max_steps
            timing_metrics = {f"timing/{k}": v for k, v in self._timers.items()}
            timing_metrics["timing/total"] = total
            self.logger.log_step(timing_metrics, step=step)

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

    def _set_text_state(self, text: str):
        if self.logger:
            self.logger.broadcast_text_state(text)

    def _stage_prefix(self) -> str:
        """Return 'Stage 2/5: name · ' if in a pipeline, else ''."""
        if hasattr(self, "_current_stage_index") and hasattr(self, "_current_stage_name"):
            total = getattr(self, "_total_stages", "?")
            return f"Stage {self._current_stage_index + 1}/{total}: {self._current_stage_name} · "
        return ""

    def _fmt_training_state(self, step: int, loss: float | None = None, dt: float | None = None) -> str:
        """Build a rich text state string for the dashboard."""
        tc = self.config.training
        dataset = getattr(self, "_current_dataset", self.config.data.dataset_name)
        parts: list[str] = [f"Training on {dataset}"]
        if self._max_flops is not None and self._max_flops > 0:
            pct = self.flops_total / self._max_flops * 100
            parts.append(f"{pct:.1f}% FLOPs · Step {step:,}")
        else:
            parts.append(f"Step {step:,}/{tc.max_steps:,}")
        if loss is not None:
            parts.append(f"Loss {loss:.3f}")
        if dt is not None and dt > 0:
            toks_per_step = self.config.data.batch_size * self.config.data.max_seq_len * tc.gradient_accumulation_steps
            tok_s = toks_per_step / dt
            parts.append(f"{tok_s:,.0f} tok/s")
        lr = self.scheduler.get_last_lr()[0]
        parts.append(f"LR {lr:.2e}")

        # Warmup indicator
        warmup = self.config.scheduler.warmup_steps
        relative_step = step - self.start_step
        if warmup > 0 and relative_step < warmup:
            parts.append(f"Warmup {relative_step}/{warmup}")

        return self._stage_prefix() + " · ".join(parts)

    def _compute_flops_per_token(self) -> int:
        """Estimate total FLOPs (fwd+bwd) per token from model architecture."""
        if hasattr(self.model, "estimate_flops"):
            estimate = self.model.estimate_flops(self.config.data.max_seq_len)
            return estimate.total
        return 0

    def _resolve_max_steps(self) -> int:
        """Resolve max_steps: use explicit value, or derive from max_flops budget."""
        tc = self.config.training
        if tc.max_steps > 0:
            return tc.max_steps
        # max_steps == 0: auto from max_flops
        if tc.max_flops is not None and self._flops_per_token > 0:
            tokens_per_step = (
                self.config.data.batch_size
                * self.config.data.max_seq_len
                * tc.gradient_accumulation_steps
            )
            flops_per_step = self._flops_per_token * tokens_per_step
            return int(tc.max_flops / flops_per_step) + 1
        return 1_000  # fallback

    def train(self):
        """Main training loop."""
        tc = self.config.training
        self._flops_per_token = self._compute_flops_per_token()
        max_steps = self._resolve_max_steps()
        self._max_flops = tc.max_flops
        self._fire_callback("on_train_begin")
        dataset = getattr(self, "_current_dataset", self.config.data.dataset_name)
        budget_str = f"{tc.max_flops:.2e} FLOPs" if tc.max_flops else f"{max_steps:,} steps"
        self._set_text_state(f"{self._stage_prefix()}Starting training on {dataset} · {budget_str}")
        self.model.train()

        for step in range(self.start_step, max_steps):
            if self.should_stop:
                break
            # Check FLOPs budget
            if self._max_flops is not None and self.flops_total >= self._max_flops:
                break

            self._fire_callback("on_step_begin", step=step)
            t0 = time.perf_counter()

            # --- Forward / backward with gradient accumulation ---
            accumulated_loss = 0.0

            for _ in range(tc.gradient_accumulation_steps):
                # Data loading
                t = self._tick()
                batch = self._next_batch()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                self._tock("data", t)

                if self._use_amp:
                    # Forward
                    t = self._tick()
                    with torch.autocast(self.device.type, dtype=self._amp_dtype):
                        output = self.model(**batch)
                        loss = output["loss"] if isinstance(output, dict) else output.loss
                        hidden = output.hidden_states if hasattr(output, "hidden_states") else None
                        if self.multi_token_heads is not None and hidden is not None:
                            mt_loss, _ = compute_multi_token_loss(
                                hidden, batch["labels"],
                                self.multi_token_heads, self._multi_token_weights,
                            )
                            loss = loss + mt_loss
                        if self.echo_heads is not None and hidden is not None:
                            echo_loss, echo_metrics = compute_echo_loss(
                                hidden, batch["labels"], self.echo_heads,
                            )
                            loss = loss + echo_loss
                        if self.phantom_config is not None and hidden is not None:
                            phantom_loss, _ = compute_phantom_loss(
                                hidden, batch["labels"],
                                self.model.norm_f, self.model.head,
                                self.phantom_config,
                            )
                            loss = loss + phantom_loss
                        if self.anchor_heads is not None and hidden is not None:
                            anchor_loss, _ = compute_state_anchor_loss(
                                hidden, self.anchor_heads,
                            )
                            loss = loss + anchor_loss
                        loss = loss / tc.gradient_accumulation_steps
                    self._tock("forward", t)
                    # Backward
                    t = self._tick()
                    self._scaler.scale(loss).backward()
                    self._tock("backward", t)
                else:
                    # Forward
                    t = self._tick()
                    output = self.model(**batch)
                    loss = output["loss"] if isinstance(output, dict) else output.loss
                    hidden = output.hidden_states if hasattr(output, "hidden_states") else None
                    if self.multi_token_heads is not None and hidden is not None:
                        mt_loss, _ = compute_multi_token_loss(
                            hidden, batch["labels"],
                            self.multi_token_heads, self._multi_token_weights,
                        )
                        loss = loss + mt_loss
                    if self.echo_heads is not None and hidden is not None:
                        echo_loss, echo_metrics = compute_echo_loss(
                            hidden, batch["labels"], self.echo_heads,
                        )
                        loss = loss + echo_loss
                    if self.phantom_config is not None and hidden is not None:
                        phantom_loss, _ = compute_phantom_loss(
                            hidden, batch["labels"],
                            self.model.norm_f, self.model.head,
                            self.phantom_config,
                        )
                        loss = loss + phantom_loss
                    if self.anchor_heads is not None and hidden is not None:
                        anchor_loss, _ = compute_state_anchor_loss(
                            hidden, self.anchor_heads,
                        )
                        loss = loss + anchor_loss
                    loss = loss / tc.gradient_accumulation_steps
                    self._tock("forward", t)
                    # Backward
                    t = self._tick()
                    loss.backward()
                    self._tock("backward", t)
                accumulated_loss += loss.item()

                tokens_batch = batch["input_ids"].numel()
                self.tokens_seen += tokens_batch
                self.flops_total += self._flops_per_token * tokens_batch

            # --- Optimizer (grad clip + sharpening + step + scheduler) ---
            t = self._tick()
            if self._use_amp and self._scaler is not None and self._scaler.is_enabled():
                self._scaler.unscale_(self.optimizer)
            if self.config.optimizer.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimizer.grad_clip_norm,
                ).item()
            else:
                # Still compute grad norm for logging even without clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), float('inf'),
                ).item()

            if self.grad_sharpen_keep is not None:
                sharpen_gradients(self.model, keep_ratio=self.grad_sharpen_keep)

            if self._use_amp and self._scaler is not None and self._scaler.is_enabled():
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            self._tock("optimizer", t)

            # --- Callbacks (before zero_grad so gradients are still available) ---
            t = self._tick()
            self._fire_callback("on_step_end", step=step)
            self._tock("callbacks", t)

            self.optimizer.zero_grad(set_to_none=True)

            dt = time.perf_counter() - t0

            # --- Logging ---
            if step % tc.log_interval == 0 and self.logger:
                t = self._tick()
                step_metrics = {
                    "train/loss": accumulated_loss,
                    "train/grad_norm": grad_norm,
                    "train/lr": self.scheduler.get_last_lr()[0],
                    "train/step": step,
                    "train/tokens_seen": self.tokens_seen,
                    "train/flops_total": self.flops_total,
                    "train/step_time": dt,
                }
                if self._max_flops is not None and self._max_flops > 0:
                    step_metrics["train/flops_pct"] = self.flops_total / self._max_flops * 100
                if hasattr(self, "_current_stage_index"):
                    step_metrics["stage/index"] = self._current_stage_index
                self.logger.log_step(step_metrics, step=step)
                self._set_text_state(self._fmt_training_state(step, accumulated_loss, dt))
                self._tock("logging", t)

            # --- Evaluation ---
            if step > 0 and step % tc.eval_interval == 0:
                self._set_text_state(f"{self._stage_prefix()}Evaluating · Step {step:,}/{tc.max_steps:,}")
                t = self._tick()
                val_metrics = self.evaluate()
                if self.logger:
                    self.logger.log_step(val_metrics, step=step)
                self._fire_callback("on_eval_end", step=step, metrics=val_metrics)
                self._set_text_state(self._fmt_training_state(step, accumulated_loss, dt))
                self._tock("eval", t)

                # Checkpoint
                if step % tc.save_interval == 0:
                    t = self._tick()
                    self._set_text_state(f"{self._stage_prefix()}Saving checkpoint · Step {step:,}")
                    self.checkpoint_manager.save(
                        step, self.model, self.optimizer, self.scheduler,
                        val_metrics, self.tokens_seen, self.flops_total,
                    )
                    self._tock("checkpoint", t)

        self._log_timing_summary()
        self._fire_callback("on_train_end")
        if self.logger:
            self.logger.close()

    def _get_cached_val_batches(self) -> list[dict]:
        """Cache validation batches on first call so we don't re-tokenize every eval."""
        if self._cached_val_batches is not None:
            return self._cached_val_batches

        max_batches = self.config.data.max_eval_batches
        batches = []
        for batch in self.val_loader:
            batches.append({k: v.to(self.device) for k, v in batch.items()})
            if max_batches > 0 and len(batches) >= max_batches:
                break

        self._cached_val_batches = batches
        return batches

    def evaluate(self) -> dict:
        """Run validation loop using cached validation batches."""
        self.model.eval()
        total_loss = 0.0

        val_batches = self._get_cached_val_batches()
        n = len(val_batches)

        with torch.inference_mode():
            for i, batch in enumerate(val_batches):
                if self._use_amp:
                    with torch.autocast(self.device.type, dtype=self._amp_dtype):
                        output = self.model(**batch)
                        loss = output["loss"] if isinstance(output, dict) else output.loss
                else:
                    output = self.model(**batch)
                    loss = output["loss"] if isinstance(output, dict) else output.loss
                total_loss += loss.item()
                if (i + 1) % 10 == 0:
                    avg_so_far = total_loss / (i + 1)
                    self._set_text_state(
                        f"{self._stage_prefix()}Evaluating · Batch {i+1}/{n} · Val loss {avg_so_far:.3f}"
                    )

        self.model.train()
        empty_cache(self.device)

        avg_loss = total_loss / max(n, 1)
        return {"val/loss": avg_loss}

    def resume(self, checkpoint_path: str):
        """Resume training from a checkpoint."""
        state = self.checkpoint_manager.load(checkpoint_path, self.device)
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.start_step = state["step"] + 1
        self.tokens_seen = state.get("tokens_seen", 0)
        self.flops_total = state.get("flops_total", 0)
        print(f"Resumed from step {state['step']}")
