"""Evaluate baseline HuggingFace models to establish reference numbers."""

import argparse

from src.evaluation import EvalConfig, evaluate
from src.evaluation.hf_model import HFModel
from src.storage import EvalDatabase

MODELS = {
    "smollm-135m": "HuggingFaceTB/SmolLM-135M",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
}


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        choices=list(MODELS.keys()),
        help="Which models to evaluate",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["perplexity", "blimp"],
        help="Which eval tasks to run",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--db", default="eval.db", help="Eval database path")
    args = parser.parse_args()

    db = EvalDatabase(args.db)

    for model_key in args.models:
        model_name = MODELS[model_key]
        print(f"\n{'#' * 60}")
        print(f"# {model_key} ({model_name})")
        print(f"{'#' * 60}\n")

        model = HFModel(model_name, device=args.device)

        config = EvalConfig(
            tasks=args.tasks,
            max_samples=args.max_samples,
        )

        evaluate(model, config, db=db, model_name=model_key)

        # Free memory before loading next model
        del model

    # Print comparison from DB
    print(f"\n{'=' * 60}")
    print("BASELINE COMPARISON (from DB)")
    print(f"{'=' * 60}")

    evals = db.get_evals()
    # Group by model
    by_model: dict[str, dict[str, dict]] = {}
    for e in evals:
        mn = e.get("model_name") or "unknown"
        if mn not in by_model:
            by_model[mn] = {}
        import json
        by_model[mn][e["task"]] = json.loads(e["metrics"])

    # Collect metric names
    metric_names: list[str] = []
    for model_tasks in by_model.values():
        for task, metrics in model_tasks.items():
            for m in metrics:
                key = f"{task}/{m}"
                if key not in metric_names:
                    metric_names.append(key)

    print(f"{'Model':<20s}", end="")
    for m in metric_names:
        print(f"{m:>20s}", end="")
    print()
    print("-" * (20 + 20 * len(metric_names)))

    for model_key, model_tasks in by_model.items():
        print(f"{model_key:<20s}", end="")
        for m in metric_names:
            task, metric = m.split("/", 1)
            val = model_tasks.get(task, {}).get(metric)
            if val is not None:
                print(f"{val:>20.4f}", end="")
            else:
                print(f"{'--':>20s}", end="")
        print()

    db.close()
    print(f"\nResults stored in {args.db}")


if __name__ == "__main__":
    main()
