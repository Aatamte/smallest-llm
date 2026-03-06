"""Unified evaluation suite — run subsets of benchmarks on a checkpoint.

Subsets:
  quick       — perplexity only (~30s)
  native      — all native tasks: perplexity, blimp, lambada, state_tracking, generation_quality
  grammar     — blimp only (linguistic minimal pairs)
  ssm         — state_tracking (synthetic SSM probes)
  generation  — generation_quality (repetition, diversity, structure)
  harness     — lm-evaluation-harness standard benchmarks (hellaswag, arc, piqa, winogrande, boolq)
  full        — native + harness (everything)

Usage:
  uv run python scripts/eval_suite.py <checkpoint> --subset quick
  uv run python scripts/eval_suite.py <checkpoint> --subset native
  uv run python scripts/eval_suite.py <checkpoint> --subset ssm
  uv run python scripts/eval_suite.py <checkpoint> --subset harness --limit 100
  uv run python scripts/eval_suite.py <checkpoint> --subset full
  uv run python scripts/eval_suite.py <checkpoint> --tasks perplexity state_tracking
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.evaluation.checkpoint_model import CheckpointModel
from src.evaluation.config import EvalConfig
from src.evaluation.results import EvalResult
from src.evaluation.runner import evaluate
from src.storage import EvalDatabase

# ── Subset definitions ──────────────────────────────────────────────

NATIVE_SUBSETS = {
    "quick": ["perplexity"],
    "native": ["perplexity", "blimp", "lambada", "state_tracking", "generation_quality"],
    "grammar": ["blimp"],
    "ssm": ["state_tracking"],
    "generation": ["generation_quality"],
    "language": ["perplexity", "lambada"],
}

HARNESS_SUBSETS = {
    "harness": [
        "hellaswag", "arc_easy", "arc_challenge",
        "winogrande", "piqa", "boolq",
    ],
    "harness_quick": ["hellaswag", "arc_easy", "piqa"],
    "harness_full": [
        "hellaswag", "arc_easy", "arc_challenge",
        "winogrande", "piqa", "boolq",
        "lambada_openai", "sciq",
    ],
}

COMBINED_SUBSETS = {
    "full": {
        "native": NATIVE_SUBSETS["native"],
        "harness": HARNESS_SUBSETS["harness"],
    },
}


def _run_native_tasks(
    model: CheckpointModel,
    tasks: list[str],
    max_samples: int | None,
    db_path: str | None,
    run_id: int | None,
    step: int | None,
    model_name: str | None,
) -> dict[str, EvalResult]:
    """Run native evaluation tasks."""
    config = EvalConfig(
        tasks=tasks,
        max_samples=max_samples,
    )
    db = EvalDatabase(db_path) if db_path else None
    results = evaluate(
        model,
        config=config,
        db=db,
        run_id=run_id,
        step=step,
        model_name=model_name,
    )
    if db:
        db.close()
    return results


def _run_harness_tasks(
    model: CheckpointModel,
    tasks: list[str],
    limit: int | None,
    num_fewshot: int,
    batch_size: int,
    db_path: str | None,
    model_name: str | None,
) -> dict[str, EvalResult]:
    """Run lm-evaluation-harness tasks."""
    try:
        import lm_eval
    except ImportError:
        print("ERROR: lm-evaluation-harness not installed.")
        print("  Install with: uv pip install lm-eval")
        return {}

    from src.evaluation.lm_harness_adapter import CheckpointLMAdapter
    from src.evaluation.lm_harness_results import (
        harness_results_to_eval_results,
        persist_harness_results,
    )

    adapter = CheckpointLMAdapter(checkpoint_model=model, batch_size=batch_size)

    print(f"\n{'=' * 60}")
    print(f"HARNESS EVALUATION")
    print(f"Tasks: {', '.join(tasks)}")
    if limit:
        print(f"Limit: {limit} samples per task")
    print(f"{'=' * 60}\n")

    results = lm_eval.simple_evaluate(
        model=adapter,
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        limit=limit,
        log_samples=False,
    )

    if results is None:
        print("No results returned from harness.")
        return {}

    eval_results = harness_results_to_eval_results(results)

    # Print results
    print(f"\n{'=' * 60}")
    print("HARNESS RESULTS")
    print(f"{'=' * 60}")
    for result in eval_results.values():
        if result.metrics:
            print(result.summary_line())
    print(f"{'=' * 60}\n")

    # Persist
    if db_path:
        db = EvalDatabase(db_path)
        persist_harness_results(eval_results, db, model_name=model_name)
        db.close()
        print(f"Harness results stored in {db_path}")

    return eval_results


def _print_report(
    native_results: dict[str, EvalResult],
    harness_results: dict[str, EvalResult],
    elapsed: float,
):
    """Print a combined summary report."""
    print(f"\n{'═' * 70}")
    print(f"  EVALUATION REPORT")
    print(f"{'═' * 70}")

    if native_results:
        print(f"\n  Native Tasks:")
        print(f"  {'─' * 60}")
        for result in native_results.values():
            if result.metrics:
                for k, v in result.metrics.items():
                    print(f"    {result.task_name}/{k}: {v}")

    if harness_results:
        print(f"\n  Harness Benchmarks:")
        print(f"  {'─' * 60}")
        for result in harness_results.values():
            if result.metrics:
                # Show acc or acc_norm as the primary metric
                acc = result.metrics.get("acc_norm", result.metrics.get("acc", None))
                if acc is not None:
                    name = result.task_name.replace("harness/", "")
                    print(f"    {name}: {acc:.4f}")
                else:
                    for k, v in result.metrics.items():
                        name = result.task_name.replace("harness/", "")
                        print(f"    {name}/{k}: {v}")

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"{'═' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Run evaluation suite on a checkpoint")
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument(
        "--subset", default="native",
        help=f"Eval subset: {', '.join(list(NATIVE_SUBSETS) + list(HARNESS_SUBSETS) + list(COMBINED_SUBSETS))}",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None,
        help="Override: specific native task names to run",
    )
    parser.add_argument(
        "--harness-tasks", nargs="+", default=None,
        help="Override: specific harness task names to run",
    )
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples for native tasks")
    parser.add_argument("--limit", type=int, default=None, help="Max samples for harness tasks")
    parser.add_argument("--num-fewshot", type=int, default=0, help="Few-shot examples for harness")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for harness")
    parser.add_argument("--db", default="eval.db", help="Eval database path (empty string to skip)")
    parser.add_argument("--model-name", default=None, help="Model name for DB storage")
    parser.add_argument("--run-id", type=int, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--save-json", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    t0 = time.perf_counter()

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    model = CheckpointModel(args.checkpoint)
    model_name = args.model_name or f"checkpoint:{Path(args.checkpoint).stem}"

    db_path = args.db if args.db else None

    # Determine what to run
    native_tasks = []
    harness_tasks = []

    if args.tasks:
        # Explicit native task override
        native_tasks = args.tasks
    if args.harness_tasks:
        # Explicit harness task override
        harness_tasks = args.harness_tasks

    if not args.tasks and not args.harness_tasks:
        # Use subset
        subset = args.subset
        if subset in NATIVE_SUBSETS:
            native_tasks = NATIVE_SUBSETS[subset]
        elif subset in HARNESS_SUBSETS:
            harness_tasks = HARNESS_SUBSETS[subset]
        elif subset in COMBINED_SUBSETS:
            combo = COMBINED_SUBSETS[subset]
            native_tasks = combo.get("native", [])
            harness_tasks = combo.get("harness", [])
        else:
            print(f"Unknown subset: {subset}")
            print(f"Available: {', '.join(list(NATIVE_SUBSETS) + list(HARNESS_SUBSETS) + list(COMBINED_SUBSETS))}")
            sys.exit(1)

    # Run native tasks
    native_results: dict[str, EvalResult] = {}
    if native_tasks:
        native_results = _run_native_tasks(
            model, native_tasks, args.max_samples,
            db_path, args.run_id, args.step, model_name,
        )

    # Run harness tasks
    harness_results: dict[str, EvalResult] = {}
    if harness_tasks:
        harness_results = _run_harness_tasks(
            model, harness_tasks, args.limit, args.num_fewshot,
            args.batch_size, db_path, model_name,
        )

    elapsed = time.perf_counter() - t0

    # Combined report
    _print_report(native_results, harness_results, elapsed)

    # Save JSON if requested
    if args.save_json:
        all_results = {}
        for name, result in {**native_results, **harness_results}.items():
            all_results[name] = result.to_dict()
        Path(args.save_json).write_text(json.dumps(all_results, indent=2))
        print(f"Results saved to {args.save_json}")

    del model


if __name__ == "__main__":
    main()
