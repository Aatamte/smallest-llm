"""ARC evaluation — AI2 Reasoning Challenge.

Multiple-choice science questions at grade-school level.
Two subsets: ARC-Easy and ARC-Challenge.
Uses loglikelihood scoring: P(answer | question) for each choice.
"""
from __future__ import annotations

import time
from typing import Optional

from src.evaluation.config import EvalConfig
from src.evaluation.protocol import Evaluatable
from src.evaluation.results import EvalResult
from src.evaluation.tasks import register_task
from src.evaluation.tasks.base import EvalTask, ProgressCallback


def _evaluate_arc(
    model: Evaluatable, config: EvalConfig, subset: str,
    task_name: str, on_progress: Optional[ProgressCallback] = None,
) -> EvalResult:
    from src.evaluation.tasks.dataset_cache import load_cached_dataset
    ds = load_cached_dataset("allenai/ai2_arc", subset=subset, split="test", cache_dir=config.data_dir)

    if config.max_samples is not None:
        ds = ds.select(range(min(config.max_samples, len(ds))))

    total = len(ds)
    if on_progress:
        on_progress(0, total)

    t0 = time.perf_counter()
    correct = 0
    tokenizer = model.tokenizer

    for i, sample in enumerate(ds):
        question = sample["question"]
        choices = sample["choices"]["text"]
        labels = sample["choices"]["label"]
        answer_key = sample["answerKey"]

        gold_idx = labels.index(answer_key) if answer_key in labels else 0

        ctx_ids = tokenizer.encode(f"Question: {question}\nAnswer:")
        choice_ids = [tokenizer.encode(" " + c) for c in choices]

        if hasattr(model, "loglikelihood_choices"):
            scores = model.loglikelihood_choices(ctx_ids, choice_ids)
        else:
            scores = [model.loglikelihood(ctx_ids, c) for c in choice_ids]

        predicted = max(range(len(scores)), key=lambda j: scores[j])
        if predicted == gold_idx:
            correct += 1

        if on_progress:
            on_progress(i + 1, total)

    accuracy = correct / total if total > 0 else 0.0
    elapsed = time.perf_counter() - t0

    return EvalResult(
        task_name=task_name,
        metrics={"accuracy": round(accuracy, 4)},
        metadata={
            "num_samples": total,
            "correct": correct,
            "elapsed_seconds": round(elapsed, 2),
        },
    )


@register_task("arc_easy")
class ArcEasyTask(EvalTask):
    name = "arc_easy"

    def download(self, data_dir: str) -> None:
        from src.evaluation.tasks.dataset_cache import load_cached_dataset
        load_cached_dataset("allenai/ai2_arc", subset="ARC-Easy", split="test", cache_dir=data_dir)

    def evaluate(self, model: Evaluatable, config: EvalConfig,
                 on_progress: Optional[ProgressCallback] = None) -> EvalResult:
        return _evaluate_arc(model, config, "ARC-Easy", self.name, on_progress)


@register_task("arc_challenge")
class ArcChallengeTask(EvalTask):
    name = "arc_challenge"

    def download(self, data_dir: str) -> None:
        from src.evaluation.tasks.dataset_cache import load_cached_dataset
        load_cached_dataset("allenai/ai2_arc", subset="ARC-Challenge", split="test", cache_dir=data_dir)

    def evaluate(self, model: Evaluatable, config: EvalConfig,
                 on_progress: Optional[ProgressCallback] = None) -> EvalResult:
        return _evaluate_arc(model, config, "ARC-Challenge", self.name, on_progress)
