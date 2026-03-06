"""Centralized run lifecycle management."""

from __future__ import annotations

import json
import os
import random
import shutil
import threading
import traceback
from datetime import date
from typing import TYPE_CHECKING, Any

from src.server.broadcast import Broadcaster
from src.storage import MainDatabase
from src.types.status import DBRunStatus

if TYPE_CHECKING:
    from src.training.trainer import Trainer

# Word lists for random name generation
_ADJECTIVES = [
    "amber", "bold", "calm", "dark", "eager", "fast", "green", "happy",
    "idle", "keen", "loud", "mild", "neat", "odd", "pale", "quick",
    "red", "slim", "tall", "vast", "warm", "young", "zinc", "blue",
    "cool", "deep", "fair", "gold", "iron", "jade", "kind", "lean",
    "nova", "opal", "pure", "rare", "sage", "true", "wild", "zero",
]

_NOUNS = [
    "ant", "bat", "cat", "dog", "elk", "fox", "gnu", "hawk",
    "ibis", "jay", "koi", "lynx", "moth", "newt", "owl", "puma",
    "quail", "ram", "seal", "toad", "urchin", "vole", "wasp", "yak",
    "bear", "crow", "dove", "frog", "goat", "hare", "lark", "mole",
    "orca", "pike", "ray", "swan", "wren", "wolf", "crab", "dart",
]


def _generate_base_name(arch: str, config_name: str) -> str:
    """Generate a base run name like 'transformer-bold-fox-20260301'."""
    if config_name == "default":
        name_part = f"{random.choice(_ADJECTIVES)}-{random.choice(_NOUNS)}"
    else:
        name_part = config_name

    date_str = date.today().strftime("%Y%m%d")
    return f"{arch}-{name_part}-{date_str}"


def _format_param_count(n: int) -> str:
    """Format parameter count: 0.5M, 1.2B, 128K, etc."""
    if n >= 1_000_000_000:
        return f"{n / 1e9:.1f}B"
    elif n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    elif n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


class RunManager:
    """Single authority for starting, stopping, and tracking training runs.

    Owns the shared Database instance and manages the active training thread.
    """

    def __init__(self, db_path: str, broadcaster: Broadcaster):
        self.db = MainDatabase(db_path)
        self.db.set_broadcaster(broadcaster)
        self.broadcaster = broadcaster
        self._lock = threading.Lock()
        self._active_run_id: int | None = None
        self._active_trainer: Trainer | None = None
        self._active_thread: threading.Thread | None = None
        self._stop_requested = threading.Event()

    # ── Queries ────────────────────────────────────────────

    def get_active(self) -> dict[str, Any] | None:
        """Return the currently active run, or None."""
        with self._lock:
            if self._active_run_id is None:
                return None
            return {
                "run_id": self._active_run_id,
                "status": "running",
            }

    # ── Start ──────────────────────────────────────────────

    def start(self, config=None) -> int:
        """Start a new training run. Returns the run_id."""
        from src.config.base import ExperimentConfig
        from src.training.run import build_pipeline_runner, build_trainer
        from src.utils.env import get_env_info

        if config is None:
            config = ExperimentConfig()

        with self._lock:
            if self._active_thread is not None and self._active_thread.is_alive():
                raise RuntimeError("A training run is already in progress")

        # Generate base name (without param count)
        base_name = _generate_base_name(config.model.name, config.name)

        # Create DB record
        run_id = self.db.create_run(base_name, config.to_dict(), get_env_info())

        # Build trainer (and optionally pipeline runner for staged training)
        pipeline_runner = None
        if config.stages:
            pipeline_runner, trainer, run_id = build_pipeline_runner(
                config=config,
                db=self.db,
                broadcaster=self.broadcaster,
                run_id=run_id,
                checkpoint_db=self.db,
            )
        else:
            trainer, run_id = build_trainer(
                config=config,
                db=self.db,
                broadcaster=self.broadcaster,
                run_id=run_id,
                checkpoint_db=self.db,
            )

        # Append param count to the name now that model is built
        param_count = trainer.model.count_parameters()
        full_name = f"{base_name}-{_format_param_count(param_count)}"
        self.db.rename_run(run_id, full_name)

        self._stop_requested.clear()

        def _run():
            try:
                trainer.logger.broadcast_status("training")
                if pipeline_runner is not None:
                    pipeline_runner.run()
                else:
                    trainer.train()
                # Only mark completed if stop wasn't requested
                if not self._stop_requested.is_set():
                    # Export HF-format model
                    try:
                        from src.training.export import export_model
                        run = self.db.get_run(run_id)
                        run_name = run["name"] if run else f"run-{run_id}"
                        model_dir = os.path.join("models", run_name)
                        export_model(
                            trainer.model, trainer.tokenizer,
                            trainer.config, model_dir,
                        )
                        self.db.create_model(
                            run_id=run_id, name=run_name, path=model_dir,
                            config=trainer.config.to_dict(),
                        )
                        print(f"Exported model to {model_dir}")
                    except Exception:
                        traceback.print_exc()
                        print("Warning: model export failed, training data is preserved")

                    self.db.finish_run(run_id, status="completed")
                    self.db.set_live_status(run_id, "complete")
            except Exception:
                traceback.print_exc()
                if not self._stop_requested.is_set():
                    self.db.finish_run(run_id, status="failed")
                self.db.set_live_status(run_id, "idle")
            finally:
                with self._lock:
                    self._active_run_id = None
                    self._active_trainer = None
                    self._active_thread = None

        thread = threading.Thread(target=_run, daemon=True)

        with self._lock:
            self._active_run_id = run_id
            self._active_trainer = trainer
            self._active_thread = thread

        thread.start()

        return run_id

    # ── Stop ───────────────────────────────────────────────

    def stop(self, run_id: int, timeout: float = 30.0) -> bool:
        """Gracefully stop the given run. Returns True if stopped."""
        with self._lock:
            if self._active_run_id != run_id:
                raise ValueError(
                    f"Run {run_id} is not the active run"
                    + (f" (active: {self._active_run_id})" if self._active_run_id else " (no active run)")
                )
            trainer = self._active_trainer
            thread = self._active_thread

        if trainer is None or thread is None:
            return False

        # Set flag BEFORE signaling trainer so _run() won't overwrite DB status
        self._stop_requested.set()
        trainer.should_stop = True
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Thread didn't stop in time — force-mark in DB
            if run_id is not None:
                self.db.finish_run(run_id, status="failed")
            return False

        # Thread exited cleanly — ensure DB status is 'stopped'
        if run_id is not None:
            self.db.finish_run(run_id, status="stopped")
            self.db.set_live_status(run_id, "idle")

        return True

    # ── Delete ──────────────────────────────────────────────

    def delete(self, run_id: int):
        """Delete a run and all its data. Stops it first if active.

        Does NOT delete the exported model — models are independent of runs.
        """
        with self._lock:
            is_active = self._active_run_id == run_id
        if is_active:
            self.stop(run_id)

        # Clean up checkpoint files on disk
        run = self.db.get_run(run_id)
        if run:
            config = json.loads(run.get("config", "{}"))
            checkpoint_dir = os.path.join(
                config.get("checkpoint", {}).get("save_dir", "checkpoints"),
                run["name"],
            )
            if os.path.isdir(checkpoint_dir):
                shutil.rmtree(checkpoint_dir)

        self.db.delete_run(run_id)

    # ── Lifecycle ──────────────────────────────────────────

    def recover_stale(self) -> list[int]:
        """Mark any runs left as 'running' from a previous crash as 'failed'."""
        stale = self.db.mark_stale_runs()
        if stale:
            print(f"Recovered {len(stale)} stale run(s): {stale}")
        return stale

    def shutdown(self):
        """Graceful server shutdown — stop active run, close DB."""
        active = self.get_active()
        if active:
            print(f"Shutting down: stopping active run #{active['run_id']}...")
            self.stop(active["run_id"], timeout=10.0)
        self.db.close()
