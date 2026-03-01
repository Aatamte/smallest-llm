"""Perplexity and bits-per-character evaluation."""

from __future__ import annotations

import math
import time
from pathlib import Path

import requests

from src.evaluation.config import EvalConfig
from src.evaluation.protocol import Evaluatable
from src.evaluation.results import EvalResult
from src.evaluation.tasks import register_task
from src.evaluation.tasks.base import EvalTask

# WikiText-2 raw test set URL
_WIKITEXT2_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/"
    "main/word_language_model/data/wikitext-2/test.txt"
)


@register_task("perplexity")
class PerplexityTask(EvalTask):
    """Compute perplexity and BPC on held-out text.

    Supports WikiText-2 (auto-downloaded) or an arbitrary text file
    passed via config.
    """

    name = "perplexity"

    def download(self, data_dir: str) -> None:
        dest = Path(data_dir) / "wikitext2_test.txt"
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(_WIKITEXT2_URL, timeout=30)
        resp.raise_for_status()
        dest.write_text(resp.text, encoding="utf-8")

    def evaluate(self, model: Evaluatable, config: EvalConfig, on_progress=None) -> EvalResult:
        data_path = Path(config.data_dir) / "wikitext2_test.txt"
        self.download(config.data_dir)

        text = data_path.read_text(encoding="utf-8").strip()
        if not text:
            raise RuntimeError(f"Empty text file: {data_path}")

        tokenizer = model.tokenizer
        token_ids = tokenizer.encode(text)

        if config.max_samples is not None:
            token_ids = token_ids[: config.max_samples]
            # Re-decode to get the corresponding character count
            text = tokenizer.decode(token_ids)

        num_tokens = len(token_ids)
        num_chars = len(text)

        if num_tokens < 2:
            raise RuntimeError("Need at least 2 tokens for perplexity")

        t0 = time.perf_counter()

        # Compute total log-likelihood, with optional progress reporting
        if on_progress:
            on_progress(0, num_tokens)
        total_ll = model.loglikelihood_rolling(token_ids, on_progress=on_progress)

        elapsed = time.perf_counter() - t0

        # Perplexity = exp(-avg_log_likelihood)
        avg_nll = -total_ll / num_tokens
        perplexity = math.exp(avg_nll)

        # Cross-entropy in nats
        cross_entropy = avg_nll

        # BPC = total NLL in bits / number of characters
        if num_chars > 0:
            bpc = (-total_ll / math.log(2)) / num_chars
        else:
            bpc = float("inf")

        return EvalResult(
            task_name=self.name,
            metrics={
                "perplexity": round(perplexity, 2),
                "bpc": round(bpc, 4),
                "cross_entropy_nats": round(cross_entropy, 4),
            },
            metadata={
                "num_tokens": num_tokens,
                "num_chars": num_chars,
                "elapsed_seconds": round(elapsed, 2),
                "dataset": "wikitext2",
            },
        )
