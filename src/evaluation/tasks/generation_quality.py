"""Generation quality evaluation — measure coherence and diversity of generated text.

Tests whether the model produces readable, non-degenerate text:
- Repetition: does it get stuck in loops?
- Diversity: unique n-gram ratios
- Structure: does it produce sentence boundaries, varied punctuation?

Uses model.generate() with various prompts and analyzes the output.
"""

from __future__ import annotations

import math
import time
from collections import Counter
from typing import Optional

from src.evaluation.config import EvalConfig
from src.evaluation.protocol import Evaluatable
from src.evaluation.results import EvalResult
from src.evaluation.tasks import register_task
from src.evaluation.tasks.base import EvalTask, ProgressCallback

# Prompts that should elicit different types of text
_PROMPTS = [
    "Once upon a time",
    "The weather today is",
    "In the beginning",
    "She walked into the room and",
    "The king said to his people",
    "It was a dark and stormy",
    "The old man sat by the fire and",
    "There was once a little girl who",
    "The sun rose over the mountains",
    "He opened the door and saw",
    "Long ago in a land far away",
    "The forest was quiet until",
    "My dear friend, I must tell you",
    "The ship sailed across the",
    "When morning came, the village",
    "The children played in the",
    "A wise man once said that",
    "Deep in the cave there was",
    "The river flowed gently through",
    "At the end of the road stood",
]


def _unique_ngram_ratio(tokens: list[str], n: int) -> float:
    """Fraction of unique n-grams out of total n-grams."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def _longest_repeated_ngram(tokens: list[str], max_n: int = 20) -> int:
    """Find the longest n-gram that appears consecutively repeated."""
    longest = 0
    for n in range(1, min(max_n, len(tokens) // 2) + 1):
        for i in range(len(tokens) - 2 * n + 1):
            if tokens[i:i + n] == tokens[i + n:i + 2 * n]:
                longest = max(longest, n)
                break
    return longest


def _repetition_score(text: str) -> float:
    """Score from 0 (no repetition) to 1 (fully repetitive).

    Based on compression ratio: repetitive text compresses well.
    """
    if len(text) < 10:
        return 0.0
    # Simple byte-level compression ratio using unique/total chars
    chars = list(text)
    unique_chars = len(set(chars))
    # Use character-level n-gram diversity
    bigram_div = _unique_ngram_ratio(chars, 2)
    trigram_div = _unique_ngram_ratio(chars, 3)
    # Low diversity = high repetition
    return 1.0 - (bigram_div + trigram_div) / 2


def _has_sentence_boundaries(text: str) -> bool:
    """Check if text contains sentence-ending punctuation."""
    return any(c in text for c in ".!?")


def _count_sentence_boundaries(text: str) -> int:
    """Count sentence-ending punctuation marks."""
    return sum(1 for c in text if c in ".!?")


@register_task("generation_quality")
class GenerationQualityTask(EvalTask):
    """Evaluate the quality and diversity of model-generated text."""

    name = "generation_quality"

    def download(self, data_dir: str) -> None:
        pass  # No data needed

    def evaluate(
        self,
        model: Evaluatable,
        config: EvalConfig,
        on_progress: Optional[ProgressCallback] = None,
    ) -> EvalResult:
        tokenizer = model.tokenizer
        max_prompts = config.max_samples or len(_PROMPTS)
        prompts = _PROMPTS[:max_prompts]

        total = len(prompts)
        if on_progress:
            on_progress(0, total)

        t0 = time.perf_counter()

        gen_tokens = 200  # generate 200 tokens per prompt
        all_texts: list[str] = []
        per_sample: list[dict] = []

        # Metrics accumulators
        unique_unigrams: list[float] = []
        unique_bigrams: list[float] = []
        unique_trigrams: list[float] = []
        repetition_scores: list[float] = []
        longest_repeats: list[int] = []
        has_sentences: list[float] = []
        sentence_counts: list[int] = []

        for i, prompt in enumerate(prompts):
            prompt_ids = tokenizer.encode(prompt)
            if not prompt_ids:
                continue

            generated_ids = model.generate(
                prompt_ids,
                max_new_tokens=gen_tokens,
                temperature=0.8,  # use some temperature for diversity
            )
            generated_text = tokenizer.decode(generated_ids)
            full_text = prompt + generated_text
            all_texts.append(full_text)

            # Tokenize for n-gram analysis (word-level)
            words = generated_text.split()
            chars = list(generated_text)

            # N-gram diversity (word-level)
            if len(words) >= 1:
                unique_unigrams.append(_unique_ngram_ratio(words, 1))
            if len(words) >= 2:
                unique_bigrams.append(_unique_ngram_ratio(words, 2))
            if len(words) >= 3:
                unique_trigrams.append(_unique_ngram_ratio(words, 3))

            # Repetition
            rep_score = _repetition_score(generated_text)
            repetition_scores.append(rep_score)

            longest_rep = _longest_repeated_ngram(words)
            longest_repeats.append(longest_rep)

            # Structure
            has_sent = _has_sentence_boundaries(generated_text)
            has_sentences.append(1.0 if has_sent else 0.0)
            sentence_counts.append(_count_sentence_boundaries(generated_text))

            per_sample.append({
                "prompt": prompt,
                "generated": generated_text[:200],  # truncate for readability
                "word_count": len(words),
                "unique_bigram_ratio": round(_unique_ngram_ratio(words, 2), 3) if len(words) >= 2 else 0,
                "repetition_score": round(rep_score, 3),
                "longest_repeat": longest_rep,
                "has_sentence_boundaries": has_sent,
            })

            if on_progress:
                on_progress(i + 1, total)

        elapsed = time.perf_counter() - t0

        # Aggregate metrics
        def _mean(vals):
            return sum(vals) / len(vals) if vals else 0.0

        metrics: dict[str, float] = {
            "unique_unigram_ratio": round(_mean(unique_unigrams), 4),
            "unique_bigram_ratio": round(_mean(unique_bigrams), 4),
            "unique_trigram_ratio": round(_mean(unique_trigrams), 4),
            "repetition_score": round(_mean(repetition_scores), 4),
            "max_longest_repeat": max(longest_repeats) if longest_repeats else 0,
            "avg_longest_repeat": round(_mean(longest_repeats), 2),
            "pct_with_sentences": round(_mean(has_sentences), 4),
            "avg_sentence_count": round(_mean(sentence_counts), 2),
        }

        # Cross-generation diversity: unique text across all prompts
        all_words = []
        for text in all_texts:
            all_words.extend(text.split())
        if len(all_words) >= 3:
            metrics["cross_gen_trigram_diversity"] = round(
                _unique_ngram_ratio(all_words, 3), 4
            )

        return EvalResult(
            task_name=self.name,
            metrics=metrics,
            metadata={
                "num_prompts": len(prompts),
                "tokens_per_prompt": gen_tokens,
                "temperature": 0.8,
                "elapsed_seconds": round(elapsed, 2),
            },
            per_sample=per_sample,
        )
