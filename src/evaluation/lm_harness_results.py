"""Convert lm-evaluation-harness results to our EvalResult format."""

from __future__ import annotations

import math
from typing import Any

from src.evaluation.results import EvalResult
from src.storage import EvalDatabase


def harness_results_to_eval_results(
    harness_output: dict[str, Any],
) -> dict[str, EvalResult]:
    """Convert lm-eval-harness results dict to our EvalResult format.

    Task names are prefixed with ``harness/`` to distinguish from native tasks.
    Metric keys like ``acc,none`` are cleaned to ``acc``.  Stderr values go into
    metadata rather than metrics.
    """
    results_by_task: dict[str, EvalResult] = {}
    raw_results = harness_output.get("results", {})

    for task_name, task_metrics in raw_results.items():
        metrics: dict[str, float] = {}
        metadata: dict[str, Any] = {"source": "lm-evaluation-harness"}

        for key, value in task_metrics.items():
            if not isinstance(value, (int, float)):
                if key == "alias":
                    metadata["alias"] = value
                continue
            if isinstance(value, float) and math.isnan(value):
                continue

            clean_key = key.split(",")[0] if "," in key else key
            if "stderr" in clean_key:
                metadata[clean_key] = value
            else:
                metrics[clean_key] = (
                    round(value, 4) if isinstance(value, float) else value
                )

        # Attach config info
        if "config" in harness_output:
            cfg = harness_output["config"]
            for cfg_key in ("num_fewshot", "batch_size", "limit"):
                if cfg_key in cfg:
                    metadata[cfg_key] = cfg[cfg_key]

        prefixed_name = f"harness/{task_name}"
        results_by_task[prefixed_name] = EvalResult(
            task_name=prefixed_name,
            metrics=metrics,
            metadata=metadata,
        )

    return results_by_task


def persist_harness_results(
    results: dict[str, EvalResult],
    db: EvalDatabase,
    model_name: str | None = None,
    run_id: int | None = None,
    step: int | None = None,
) -> None:
    """Persist converted harness results to eval.db."""
    for result in results.values():
        db.log_eval(
            task=result.task_name,
            metrics=result.metrics,
            metadata=result.metadata,
            run_id=run_id,
            step=step,
            model_name=model_name,
        )
