"""WinoGrande evaluation — coreference resolution.

Given a sentence with a blank, pick which of two options correctly fills it.
Uses loglikelihood scoring on the full sentence with each option substituted.
"""
from __future__ import annotations

import time
from typing import Optional

from src.evaluation.config import EvalConfig
from src.evaluation.protocol import Evaluatable
from src.evaluation.results import EvalResult
from src.evaluation.tasks import register_task
from src.evaluation.tasks.base import EvalTask, ProgressCallback


@register_task("winogrande")
class WinoGrandeTask(EvalTask):
    name = "winogrande"

    def download(self, data_dir: str) -> None:
        from src.evaluation.tasks.dataset_cache import load_cached_dataset
        load_cached_dataset("allenai/winogrande", subset="winogrande_debiased",
                            split="validation", cache_dir=data_dir)

    def evaluate(
        self, model: Evaluatable, config: EvalConfig,
        on_progress: Optional[ProgressCallback] = None,
    ) -> EvalResult:
        from src.evaluation.tasks.dataset_cache import load_cached_dataset
        ds = load_cached_dataset("allenai/winogrande", subset="winogrande_debiased",
                                 split="validation", cache_dir=config.data_dir)

        if config.max_samples is not None:
            ds = ds.select(range(min(config.max_samples, len(ds))))

        total = len(ds)
        if on_progress:
            on_progress(0, total)

        t0 = time.perf_counter()
        correct = 0
        tokenizer = model.tokenizer

        for i, sample in enumerate(ds):
            sentence = sample["sentence"]
            option1 = sample["option1"]
            option2 = sample["option2"]
            label = int(sample["answer"]) - 1  # "1" or "2" -> 0 or 1

            # Split sentence at "_" and score each option as continuation
            parts = sentence.split("_")
            if len(parts) != 2:
                if on_progress:
                    on_progress(i + 1, total)
                continue

            ctx = parts[0].rstrip()
            suffix = parts[1]

            ctx_ids = tokenizer.encode(ctx)
            choice_ids = [tokenizer.encode(" " + opt + suffix) for opt in [option1, option2]]

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
