"""Background evaluation job management."""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from src.services.model_loader import is_custom_export, load_eval_model

if TYPE_CHECKING:
    from src.services.run_service import RunManager
    from src.storage import EvalDatabase

# lm-evaluation-harness benchmark presets
HARNESS_PRESETS = {
    "quick": ["hellaswag", "arc_easy", "winogrande"],
    "standard": ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa", "boolq"],
    "full": [
        "hellaswag", "arc_easy", "arc_challenge", "winogrande",
        "piqa", "boolq", "mmlu", "truthfulqa_mc2", "gsm8k",
    ],
}
HARNESS_ALL_TASKS = HARNESS_PRESETS["full"]

_IDLE_STATE = dict(
    status="idle", model_name=None,
    stage=None, stage_index=0, stage_count=0,
    task=None, task_index=0, task_count=0,
    current_sample=0, total_samples=0,
    started_at=None, error=None,
)


class EvalService:
    """Manages background evaluation jobs."""

    def __init__(self, eval_db: EvalDatabase, run_manager: RunManager):
        self.eval_db = eval_db
        self.run_manager = run_manager
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._state: dict[str, Any] = dict(_IDLE_STATE)

    def get_status(self) -> dict:
        with self._lock:
            return dict(self._state)

    def stop(self) -> dict:
        self._stop.set()
        with self._lock:
            if self._state["status"] == "running":
                self._state.update(**_IDLE_STATE, status="stopped")
            return dict(self._state)

    def start(
        self,
        *,
        source: str,
        tasks: list[str],
        harness_tasks: list[str],
        max_samples: int | None = None,
        harness_limit: float | int | None = None,
        # For source="hf"
        model_name: str | None = None,
        # For source="checkpoint"
        run_id: int | None = None,
        step: int | None = None,
    ) -> dict:
        """Start an eval job in a background thread. Returns initial status."""

        # Resolve model
        if source == "checkpoint":
            checkpoints = self.run_manager.db.get_checkpoints(run_id)
            match = next((c for c in checkpoints if c["step"] == step), None)
            if match is None:
                raise ValueError(f"No checkpoint at step {step} for run {run_id}")

            run = self.run_manager.db.get_run(run_id)
            run_name = run["name"] if run else f"run-{run_id}"
            display_name = f"{run_name} (step {step})"
            checkpoint_path = match["path"]
            model_path = None

        elif source == "hf":
            model_row = self.run_manager.db.get_model_by_name(model_name)
            if model_row is None:
                available = [m["name"] for m in self.run_manager.db.list_models()]
                raise ValueError(f"Unknown model: {model_name}. Available: {available}")
            model_path = model_row["path"]
            display_name = model_name
            checkpoint_path = None

        else:
            raise ValueError(f"Unknown source: {source}")

        stage_count = (1 if tasks else 0) + (1 if harness_tasks else 0)

        with self._lock:
            if self._state["status"] == "running":
                raise RuntimeError("An evaluation is already running")
            self._state.update(
                status="running", model_name=display_name,
                stage="native" if tasks else "harness",
                stage_index=0, stage_count=stage_count,
                task=None, task_index=0,
                task_count=len(tasks) if tasks else len(harness_tasks),
                current_sample=0, total_samples=0,
                started_at=time.time(), error=None,
            )

        eval_db = self.eval_db

        def _progress_cb(task_index, task_count, task_name, current, total):
            if self._stop.is_set():
                from src.evaluation.runner import EvalCancelled
                raise EvalCancelled("Evaluation stopped by user")
            with self._lock:
                self._state.update(
                    stage="native", task=task_name,
                    task_index=task_index, task_count=task_count,
                    current_sample=current, total_samples=total,
                )

        def _run():
            self._stop.clear()
            try:
                # --- Phase 1: Native tasks ---
                if tasks:
                    from src.evaluation import EvalConfig, evaluate

                    if source == "checkpoint":
                        from src.evaluation.checkpoint_model import CheckpointModel
                        native_model = CheckpointModel(checkpoint_path)
                    else:
                        native_model = load_eval_model(model_path)

                    config = EvalConfig(tasks=tasks, max_samples=max_samples)
                    evaluate(native_model, config, db=eval_db, model_name=display_name, on_progress=_progress_cb)
                    del native_model

                # --- Phase 2: Harness tasks ---
                if harness_tasks:
                    self._run_harness(
                        source=source,
                        harness_tasks=harness_tasks,
                        harness_limit=harness_limit,
                        display_name=display_name,
                        checkpoint_path=checkpoint_path,
                        model_path=model_path,
                        harness_stage_index=1 if tasks else 0,
                    )

                with self._lock:
                    self._state.update(**_IDLE_STATE)

            except Exception as e:
                from src.evaluation.runner import EvalCancelled
                if isinstance(e, EvalCancelled):
                    with self._lock:
                        self._state.update(**_IDLE_STATE, status="stopped")
                else:
                    import traceback
                    traceback.print_exc()
                    with self._lock:
                        self._state.update(status="error", error=str(e))

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        return {"status": "started", "model_name": display_name, "tasks": tasks, "harness_tasks": harness_tasks}

    def _run_harness(
        self,
        *,
        source: str,
        harness_tasks: list[str],
        harness_limit: float | int | None,
        display_name: str,
        checkpoint_path: str | None,
        model_path: str | None,
        harness_stage_index: int,
    ):
        """Run lm-evaluation-harness benchmarks."""
        if self._stop.is_set():
            from src.evaluation.runner import EvalCancelled
            raise EvalCancelled("Evaluation stopped by user")

        import lm_eval
        from src.evaluation.lm_harness_adapter import CheckpointLMAdapter, ProgressLMWrapper
        from src.evaluation.lm_harness_results import harness_results_to_eval_results, persist_harness_results

        def _harness_progress(current, total):
            if self._stop.is_set():
                from src.evaluation.runner import EvalCancelled
                raise EvalCancelled("Evaluation stopped by user")
            with self._lock:
                self._state.update(
                    stage="harness",
                    stage_index=harness_stage_index,
                    task="harness benchmarks",
                    task_index=0,
                    task_count=len(harness_tasks),
                    current_sample=current,
                    total_samples=total,
                )

        if source == "checkpoint":
            from src.evaluation.checkpoint_model import CheckpointModel

            ckpt_model = CheckpointModel(checkpoint_path)
            adapter = CheckpointLMAdapter(
                checkpoint_model=ckpt_model, batch_size=4, on_progress=_harness_progress,
            )
            harness_results = lm_eval.simple_evaluate(
                model=adapter, tasks=harness_tasks, batch_size=4,
                limit=harness_limit, log_samples=False,
            )
            del ckpt_model, adapter

        elif model_path and is_custom_export(model_path):
            from src.evaluation.exported_model import ExportedModel

            exported = ExportedModel(model_path)
            adapter = CheckpointLMAdapter(
                checkpoint_model=exported, batch_size=4, on_progress=_harness_progress,
            )
            harness_results = lm_eval.simple_evaluate(
                model=adapter, tasks=harness_tasks, batch_size=4,
                limit=harness_limit, log_samples=False,
            )
            del exported, adapter

        else:
            from lm_eval.models.huggingface import HFLM

            hflm = HFLM(pretrained=model_path, dtype="float32", device="cpu", batch_size=4)
            wrapped = ProgressLMWrapper(hflm, on_progress=_harness_progress, chunk_size=16)
            harness_results = lm_eval.simple_evaluate(
                model=wrapped, tasks=harness_tasks, batch_size=4,
                limit=harness_limit, log_samples=False,
            )
            del hflm, wrapped

        if harness_results is not None:
            eval_results = harness_results_to_eval_results(harness_results)
            persist_harness_results(eval_results, self.eval_db, model_name=display_name)
