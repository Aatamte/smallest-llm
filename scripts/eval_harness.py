"""Run lm-evaluation-harness benchmarks on checkpoints and HF baselines."""

import argparse

import lm_eval

from src.evaluation.checkpoint_model import CheckpointModel
from src.evaluation.lm_harness_adapter import CheckpointLMAdapter
from src.evaluation.lm_harness_results import (
    harness_results_to_eval_results,
    persist_harness_results,
)
from src.storage import EvalDatabase

# Benchmark presets (task names match lm-evaluation-harness)
BENCHMARK_PRESETS = {
    "quick": ["hellaswag", "arc_easy", "winogrande"],
    "standard": [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "piqa",
        "boolq",
    ],
    "full": [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "piqa",
        "boolq",
        "mmlu",
        "truthfulqa_mc2",
        "gsm8k",
    ],
}

HF_BASELINES = {
    "smollm-135m": "HuggingFaceTB/SmolLM-135M",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
}


def _resolve_tasks(task_input: list[str]) -> list[str]:
    """Resolve preset names or pass through individual task names."""
    tasks = []
    for t in task_input:
        if t in BENCHMARK_PRESETS:
            tasks.extend(BENCHMARK_PRESETS[t])
        else:
            tasks.append(t)
    return list(dict.fromkeys(tasks))  # deduplicate, preserve order


def _print_results(eval_results: dict) -> None:
    """Print a summary table of results."""
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    for result in eval_results.values():
        if result.metrics:
            print(result.summary_line())
    print(f"{'=' * 60}\n")


def _run_and_store(adapter, tasks, args, model_name):
    """Run simple_evaluate, convert results, persist."""
    results = lm_eval.simple_evaluate(
        model=adapter,
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
        log_samples=False,
    )

    if results is None:
        print("No results returned.")
        return

    eval_results = harness_results_to_eval_results(results)
    _print_results(eval_results)

    if args.db:
        db = EvalDatabase(args.db)
        persist_harness_results(
            eval_results,
            db,
            model_name=model_name,
            run_id=getattr(args, "run_id", None),
            step=getattr(args, "step", None),
        )
        db.close()
        print(f"Results stored in {args.db}")


def eval_checkpoint(args):
    """Evaluate a trained checkpoint."""
    model = CheckpointModel(args.checkpoint)
    adapter = CheckpointLMAdapter(
        checkpoint_model=model, batch_size=args.batch_size
    )
    tasks = _resolve_tasks(args.tasks)
    model_name = args.model_name or f"checkpoint:{args.checkpoint.split('/')[-1]}"

    print(f"\n{'#' * 60}")
    print(f"# {model_name}")
    print(f"# Tasks: {', '.join(tasks)}")
    print(f"{'#' * 60}\n")

    _run_and_store(adapter, tasks, args, model_name)

    del model


def eval_hf(args):
    """Evaluate HF models using lm-eval's built-in hf support."""
    tasks = _resolve_tasks(args.tasks)

    for model_key in args.models:
        hf_name = HF_BASELINES[model_key]

        print(f"\n{'#' * 60}")
        print(f"# {model_key} ({hf_name})")
        print(f"# Tasks: {', '.join(tasks)}")
        print(f"{'#' * 60}\n")

        # Use lm-eval's built-in HF support directly
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={hf_name},dtype=float32,device={args.device}",
            tasks=tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
            log_samples=False,
        )

        if results is None:
            print(f"No results for {model_key}.")
            continue

        eval_results = harness_results_to_eval_results(results)
        _print_results(eval_results)

        if args.db:
            db = EvalDatabase(args.db)
            persist_harness_results(eval_results, db, model_name=model_key)
            db.close()
            print(f"Results stored in {args.db}")


def main():
    parser = argparse.ArgumentParser(
        description="Run lm-evaluation-harness benchmarks"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Subcommand: checkpoint
    cp = sub.add_parser("checkpoint", help="Evaluate a trained checkpoint")
    cp.add_argument("checkpoint", help="Path to checkpoint .pt file")
    cp.add_argument("--model-name", help="Name for DB storage")
    cp.add_argument("--run-id", type=int, default=None)
    cp.add_argument("--step", type=int, default=None)

    # Subcommand: hf
    hf = sub.add_parser("hf", help="Evaluate HF baseline models")
    hf.add_argument(
        "--models",
        nargs="+",
        default=list(HF_BASELINES.keys()),
        choices=list(HF_BASELINES.keys()),
    )

    # Shared arguments on both subcommands
    for p in [cp, hf]:
        p.add_argument(
            "--tasks",
            nargs="+",
            default=["quick"],
            help="Task names or presets: quick, standard, full",
        )
        p.add_argument("--num-fewshot", type=int, default=0)
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Max samples per task (None = full dataset)",
        )
        p.add_argument("--device", default="mps")
        p.add_argument("--db", default="eval.db", help="Eval database path")

    args = parser.parse_args()
    if args.command == "checkpoint":
        eval_checkpoint(args)
    elif args.command == "hf":
        eval_hf(args)


if __name__ == "__main__":
    main()
