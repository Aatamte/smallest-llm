"""Centralized run lifecycle management."""

from __future__ import annotations

import threading
import traceback
from typing import TYPE_CHECKING, Any

from src.server.broadcast import Broadcaster
from src.storage.database import Database
from src.types.status import DBRunStatus

if TYPE_CHECKING:
    from src.training.trainer import Trainer


class RunManager:
    """Single authority for starting, stopping, and tracking training runs.

    Owns the shared Database instance and manages the active training thread.
    """

    def __init__(self, db_path: str, broadcaster: Broadcaster):
        self.db = Database(db_path)
        self.broadcaster = broadcaster
        self._lock = threading.Lock()
        self._active_run_id: int | None = None
        self._active_trainer: Trainer | None = None
        self._active_thread: threading.Thread | None = None
        self._stop_requested: bool = False

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
        from src.training.run import build_trainer
        from src.utils.env import get_env_info

        if config is None:
            config = ExperimentConfig()

        with self._lock:
            if self._active_thread is not None and self._active_thread.is_alive():
                raise RuntimeError("A training run is already in progress")

        # Create DB record
        run_id = self.db.create_run(config.name, config.to_dict(), get_env_info())

        # Build trainer (loads data, model, etc.)
        trainer, run_id = build_trainer(
            config=config,
            db=self.db,
            broadcaster=self.broadcaster,
            run_id=run_id,
        )

        with self._lock:
            self._active_run_id = run_id
            self._active_trainer = trainer
            self._stop_requested = False

        def _run():
            try:
                trainer.logger.broadcast_status("training")
                trainer.train()
                # Only mark completed if stop wasn't requested
                if not self._stop_requested:
                    self.db.finish_run(run_id, status="completed")
                    self.broadcaster.publish({"type": "status", "data": "complete"})
            except Exception:
                traceback.print_exc()
                if not self._stop_requested:
                    self.db.finish_run(run_id, status="failed")
                self.broadcaster.publish({"type": "status", "data": "idle"})
            finally:
                with self._lock:
                    self._active_run_id = None
                    self._active_trainer = None
                    self._active_thread = None

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        with self._lock:
            self._active_thread = thread

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
        self._stop_requested = True
        trainer.should_stop = True
        thread.join(timeout=timeout)

        if thread.is_alive():
            # Thread didn't stop in time — force-mark in DB
            if run_id is not None:
                self.db.finish_run(run_id, status="failed")
            return False

        # Thread exited cleanly — the _run() finally block already cleaned up,
        # but ensure DB status is 'stopped' (not 'completed')
        if run_id is not None:
            self.db.finish_run(run_id, status="stopped")
            self.broadcaster.publish({"type": "status", "data": "idle"})

        return True

    # ── Delete ──────────────────────────────────────────────

    def delete(self, run_id: int):
        """Delete a run and all its data. Stops it first if active."""
        with self._lock:
            is_active = self._active_run_id == run_id
        if is_active:
            self.stop(run_id)
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
