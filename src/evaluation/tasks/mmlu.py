"""MMLU evaluation — Massive Multitask Language Understanding.

57 subjects across STEM, humanities, social sciences, and more.
4-way multiple choice. Uses loglikelihood scoring: P(answer | question + choices).
"""
from __future__ import annotations

import time
from typing import Optional

from src.evaluation.config import EvalConfig
from src.evaluation.protocol import Evaluatable
from src.evaluation.results import EvalResult
from src.evaluation.tasks import register_task
from src.evaluation.tasks.base import EvalTask, ProgressCallback

_CHOICE_LETTERS = ["A", "B", "C", "D"]


@register_task("mmlu")
class MMLUTask(EvalTask):
    name = "mmlu"

    def download(self, data_dir: str) -> None:
        from src.evaluation.tasks.dataset_cache import load_cached_dataset
        load_cached_dataset("cais/mmlu", subset="all", split="test", cache_dir=data_dir)

    def evaluate(
        self, model: Evaluatable, config: EvalConfig,
        on_progress: Optional[ProgressCallback] = None,
    ) -> EvalResult:
        from src.evaluation.tasks.dataset_cache import load_cached_dataset
        ds = load_cached_dataset("cais/mmlu", subset="all", split="test", cache_dir=config.data_dir)

        if config.max_samples is not None:
            ds = ds.select(range(min(config.max_samples, len(ds))))

        total = len(ds)
        if on_progress:
            on_progress(0, total)

        t0 = time.perf_counter()
        correct = 0
        per_subject: dict[str, dict[str, int]] = {}
        tokenizer = model.tokenizer

        for i, sample in enumerate(ds):
            question = sample["question"]
            choices = sample["choices"]
            label = sample["answer"]  # 0-3
            subject = sample.get("subject", "unknown")

            # Format: "Question: ... \nA. choice1\nB. choice2\n...\nAnswer:"
            formatted = f"Question: {question}\n"
            for j, choice in enumerate(choices):
                formatted += f"{_CHOICE_LETTERS[j]}. {choice}\n"
            formatted += "Answer:"

            ctx_ids = tokenizer.encode(formatted)
            choice_ids = [tokenizer.encode(" " + _CHOICE_LETTERS[j]) for j in range(len(choices))]

            if hasattr(model, "loglikelihood_choices"):
                scores = model.loglikelihood_choices(ctx_ids, choice_ids)
            else:
                scores = [model.loglikelihood(ctx_ids, c) for c in choice_ids]

            predicted = max(range(len(scores)), key=lambda j: scores[j])
            is_correct = predicted == label

            if is_correct:
                correct += 1

            # Track per-subject accuracy
            if subject not in per_subject:
                per_subject[subject] = {"correct": 0, "total": 0}
            per_subject[subject]["total"] += 1
            if is_correct:
                per_subject[subject]["correct"] += 1

            if on_progress:
                on_progress(i + 1, total)

        accuracy = correct / total if total > 0 else 0.0
        elapsed = time.perf_counter() - t0

        # Compute per-subject accuracies for metadata
        subject_accs = {
            subj: round(stats["correct"] / stats["total"], 4)
            for subj, stats in sorted(per_subject.items())
            if stats["total"] > 0
        }

        return EvalResult(
            task_name=self.name,
            metrics={"accuracy": round(accuracy, 4)},
            metadata={
                "num_samples": total,
                "correct": correct,
                "num_subjects": len(per_subject),
                "elapsed_seconds": round(elapsed, 2),
                "per_subject": subject_accs,
            },
        )
