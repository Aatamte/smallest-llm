"""Evaluation runner — orchestrates tasks and collects results."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from src.evaluation.config import EvalConfig
from src.evaluation.protocol import Evaluatable
from src.evaluation.results import EvalResult
from src.evaluation.tasks import get_task

if TYPE_CHECKING:
    from src.storage.eval_db import EvalDatabase


def evaluate(
    model: Evaluatable,
    config: EvalConfig | None = None,
    db: EvalDatabase | None = None,
    run_id: int | None = None,
    step: int | None = None,
    model_name: str | None = None,
) -> dict[str, EvalResult]:
    """Run evaluation tasks on a model.

    Args:
        model: Any object satisfying the Evaluatable protocol.
        config: Evaluation configuration. Uses defaults if None.
        db: Optional Database instance for persisting results.
        run_id: Optional run ID to associate results with.
        step: Optional training step (for mid-training evals).
        model_name: Optional model name (for baseline evals).

    Returns:
        Dict mapping task name to EvalResult.
    """
    if config is None:
        config = EvalConfig()

    results: dict[str, EvalResult] = {}

    print("=" * 60)
    print("EVALUATION")
    print("=" * 60)
    print(f"Tasks: {', '.join(config.tasks)}")
    if config.max_samples is not None:
        print(f"Max samples per task: {config.max_samples}")
    print()

    total_t0 = time.perf_counter()

    for task_name in config.tasks:
        print(f"--- {task_name} ---")
        task = get_task(task_name)

        try:
            result = task.evaluate(model, config)
            results[task_name] = result
            print(result.summary_line())

            if db is not None:
                db.log_eval(
                    task=task_name,
                    metrics=result.metrics,
                    metadata=result.metadata,
                    run_id=run_id,
                    step=step,
                    model_name=model_name,
                )
        except Exception as e:
            print(f"  ERROR: {e}")
            results[task_name] = EvalResult(
                task_name=task_name,
                metrics={},
                metadata={"error": str(e)},
            )

        print()

    total_elapsed = time.perf_counter() - total_t0

    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for result in results.values():
        if result.metrics:
            print(result.summary_line())
    print(f"\nTotal time: {total_elapsed:.1f}s")
    print("=" * 60)

    return results
