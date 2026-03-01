"""LAMBADA evaluation — predict the last word of a passage.

Tests long-range discourse understanding. The model must predict the exact
final word of each passage, which requires tracking context over 50-200+ tokens.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import requests

from src.evaluation.config import EvalConfig
from src.evaluation.protocol import Evaluatable
from src.evaluation.results import EvalResult
from src.evaluation.tasks import register_task
from src.evaluation.tasks.base import EvalTask

# LAMBADA test set (OpenAI's preprocessed version)
_LAMBADA_URL = (
    "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"
)


@register_task("lambada")
class LAMBADATask(EvalTask):
    """Evaluate last-word prediction accuracy on LAMBADA."""

    name = "lambada"

    def download(self, data_dir: str) -> None:
        dest = Path(data_dir) / "lambada_test.jsonl"
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(_LAMBADA_URL, timeout=60)
        resp.raise_for_status()
        dest.write_text(resp.text, encoding="utf-8")

    def evaluate(self, model: Evaluatable, config: EvalConfig) -> EvalResult:
        self.download(config.data_dir)
        data_path = Path(config.data_dir) / "lambada_test.jsonl"
        tokenizer = model.tokenizer

        # Load passages
        passages: list[str] = []
        with open(data_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                passages.append(obj["text"])

        if config.max_samples is not None:
            passages = passages[: config.max_samples]

        t0 = time.perf_counter()

        correct_count = 0
        total_target_ll = 0.0
        total_target_tokens = 0
        per_sample: list[dict] = []

        for passage in passages:
            # Split into context + last word
            words = passage.rsplit(" ", 1)
            if len(words) < 2:
                continue

            context_text = words[0]
            target_word = words[1]

            context_ids = tokenizer.encode(context_text + " ")
            target_ids = tokenizer.encode(target_word)

            if not context_ids or not target_ids:
                continue

            # Log-likelihood of the target word given context
            ll = model.loglikelihood(context_ids, target_ids)

            # Greedy generation check: generate same number of tokens as target
            generated_ids = model.generate(
                context_ids,
                max_new_tokens=len(target_ids),
                temperature=0.0,
            )

            # Check exact match
            predicted_text = tokenizer.decode(generated_ids).strip()
            target_text_clean = target_word.strip()
            is_correct = predicted_text == target_text_clean

            if is_correct:
                correct_count += 1

            total_target_ll += ll
            total_target_tokens += len(target_ids)

            if config.max_samples is not None and config.max_samples <= 100:
                per_sample.append({
                    "context": context_text[-100:],  # last 100 chars
                    "target": target_word,
                    "predicted": predicted_text,
                    "correct": is_correct,
                    "target_ll": round(ll, 4),
                })

        elapsed = time.perf_counter() - t0

        num_passages = len(passages)
        accuracy = correct_count / num_passages if num_passages > 0 else 0.0

        # Perplexity over target words
        if total_target_tokens > 0:
            avg_nll = -total_target_ll / total_target_tokens
            target_ppl = math.exp(min(avg_nll, 100))  # clamp to avoid overflow
        else:
            target_ppl = float("inf")

        return EvalResult(
            task_name=self.name,
            metrics={
                "accuracy": round(accuracy, 4),
                "target_perplexity": round(target_ppl, 2),
            },
            metadata={
                "num_passages": num_passages,
                "correct": correct_count,
                "elapsed_seconds": round(elapsed, 2),
            },
            per_sample=per_sample if per_sample else None,
        )
