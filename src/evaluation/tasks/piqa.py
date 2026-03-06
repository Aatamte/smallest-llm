"""PIQA evaluation — Physical Intuition QA.

Given a goal, pick which of two solutions is more physically plausible.
Uses loglikelihood scoring: P(solution | goal) for each option.
"""
from __future__ import annotations

import time
from typing import Optional

from src.evaluation.config import EvalConfig
from src.evaluation.protocol import Evaluatable
from src.evaluation.results import EvalResult
from src.evaluation.tasks import register_task
from src.evaluation.tasks.base import EvalTask, ProgressCallback


@register_task("piqa")
class PIQATask(EvalTask):
    name = "piqa"

    def download(self, data_dir: str) -> None:
        from src.evaluation.tasks.dataset_cache import load_cached_dataset
        load_cached_dataset("ybisk/piqa", split="validation", cache_dir=data_dir, revision="refs/convert/parquet")

    def evaluate(
        self, model: Evaluatable, config: EvalConfig,
        on_progress: Optional[ProgressCallback] = None,
    ) -> EvalResult:
        from src.evaluation.tasks.dataset_cache import load_cached_dataset
        ds = load_cached_dataset("ybisk/piqa", split="validation", cache_dir=config.data_dir, revision="refs/convert/parquet")

        if config.max_samples is not None:
            ds = ds.select(range(min(config.max_samples, len(ds))))

        total = len(ds)
        if on_progress:
            on_progress(0, total)

        t0 = time.perf_counter()
        correct = 0
        tokenizer = model.tokenizer

        for i, sample in enumerate(ds):
            goal = sample["goal"]
            choices = [sample["sol1"], sample["sol2"]]
            label = sample["label"]  # 0 or 1

            ctx_ids = tokenizer.encode(f"Goal: {goal}\nSolution:")
            choice_ids = [tokenizer.encode(" " + c) for c in choices]

            if hasattr(model, "loglikelihood_choices"):
                scores = model.loglikelihood_choices(ctx_ids, choice_ids)
            else:
                scores = [model.loglikelihood(ctx_ids, c) for c in choice_ids]

            predicted = max(range(len(scores)), key=lambda j: scores[j])
            if predicted == label:
                correct += 1

            if on_progress:
                on_progress(i + 1, total)

        accuracy = correct / total if total > 0 else 0.0
        elapsed = time.perf_counter() - t0

        return EvalResult(
            task_name=self.name,
            metrics={"accuracy": round(accuracy, 4)},
            metadata={
                "num_samples": total,
                "correct": correct,
                "elapsed_seconds": round(elapsed, 2),
            },
        )
