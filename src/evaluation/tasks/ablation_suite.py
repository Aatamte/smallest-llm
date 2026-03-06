"""Ablation eval suite — fast multi-tier evaluation for A/B experiments.

Covers all success tiers from goal.md in a single fast eval (~10-15s):

  Tier 1: Loss metrics (cross-entropy, perplexity, accuracy)
  Tier 2: Phrase completion ("once upon a ___" → "time")
  Tier 3: Coherence (prefer coherent vs incoherent sentences)
  Tier 4: State tracking (counting, recall, pattern completion)

Returns per-tier scores and a composite score for quick comparison.
"""

from __future__ import annotations

import math
import time
from typing import Optional

import torch
import torch.nn.functional as F

from src.evaluation.config import EvalConfig
from src.evaluation.protocol import Evaluatable
from src.evaluation.results import EvalResult
from src.evaluation.tasks import register_task
from src.evaluation.tasks.base import EvalTask, ProgressCallback
from src.evaluation.tasks.quick_loss import _HELD_OUT_TEXTS
from src.evaluation.tasks.state_tracking import (
    _generate_counting_examples,
    _generate_pattern_examples,
    _generate_recall_examples,
)


# ── Tier 2: Phrase completion prompts ────────────────────────────────────────
# Each tuple: (context, expected_next_word)
# The model should assign high probability to the expected word.

_PHRASE_COMPLETIONS = [
    # Classic phrases
    ("Once upon a ", "time"),
    ("Happily ever ", "after"),
    ("Long ago and far ", "away"),
    ("In the beginning there ", "was"),
    # Common collocations
    ("The cat sat on the ", "mat"),
    ("She went to the ", "store"),
    ("He opened the ", "door"),
    ("They walked down the ", "street"),
    ("The sun was shining ", "bright"),
    ("It was a dark and stormy ", "night"),
    # Subject-verb agreement
    ("The dog ", "was"),
    ("The children ", "were"),
    ("She ", "was"),
    ("They ", "were"),
    ("He ", "was"),
    # Articles
    ("She ate an ", "apple"),
    ("He saw a ", "bird"),
    ("There was a big ", "tree"),
    # Prepositions
    ("The bird flew over the ", "tree"),
    ("The fish swam in the ", "water"),
    ("She sat down on the ", "chair"),
    ("He looked out the ", "window"),
    ("The ball rolled under the ", "table"),
    # Verb completion
    ("The baby started to ", "cry"),
    ("She began to ", "sing"),
    ("He wanted to ", "go"),
    ("They decided to ", "leave"),
    ("The boy learned to ", "read"),
    # Common patterns
    ("Thank you very ", "much"),
    ("Good ", "morning"),
    ("How are ", "you"),
    ("I don't ", "know"),
    ("What do you ", "want"),
    # Cause and effect
    ("It was raining so she took her ", "umbrella"),
    ("He was hungry so he ate some ", "food"),
    ("She was tired so she went to ", "bed"),
    ("It was cold outside so he wore a ", "coat"),
    # Story patterns
    ("The princess lived in a ", "castle"),
    ("The king sat on his ", "throne"),
    ("The knight drew his ", "sword"),
    ("The wizard cast a ", "spell"),
    ("The dragon breathed ", "fire"),
    # Simple factual
    ("The sky is ", "blue"),
    ("Grass is ", "green"),
    ("Snow is ", "white"),
    ("The sun is ", "hot"),
    ("Water is ", "wet"),
    # Temporal
    ("In the morning she ate ", "breakfast"),
    ("At night they went to ", "sleep"),
    ("After school the children went ", "home"),
    ("Before dinner he washed his ", "hands"),
    # Emotional
    ("The funny joke made her ", "laugh"),
    ("The sad movie made him ", "cry"),
    ("The scary noise made the children ", "scream"),
    ("The good news made everyone ", "happy"),
    # Actions
    ("She picked up the ", "book"),
    ("He put on his ", "shoes"),
    ("They sat down at the ", "table"),
    ("The mother tucked her child into ", "bed"),
    ("The teacher wrote on the ", "board"),
]


# ── Tier 3: Coherence minimal pairs ─────────────────────────────────────────
# Each tuple: (coherent_sentence, incoherent_sentence)
# The model should assign higher likelihood to the coherent version.

_COHERENCE_PAIRS = [
    # Semantic coherence
    ("The dog chased the cat up the tree.",
     "The dog chased the cat up the fish."),
    ("She went to the store to buy milk.",
     "She went to the store to buy ceiling."),
    ("He put on his shoes and went outside.",
     "He put on his shoes and went purple."),
    ("The bird sat on the branch and sang.",
     "The bird sat on the branch and drove."),
    ("The baby drank milk from the bottle.",
     "The baby drank milk from the mountain."),
    # Verb agreement
    ("The children were playing in the yard.",
     "The children was playing in the yard."),
    ("She has been reading all morning.",
     "She have been reading all morning."),
    ("The cat is sleeping on the couch.",
     "The cat are sleeping on the couch."),
    # Temporal coherence
    ("First she woke up, then she ate breakfast.",
     "First she ate breakfast, then she woke up."),
    ("He finished his homework before going to play.",
     "He finished his homework before going to sleep before waking up before sleeping."),
    # Selectional restrictions
    ("The chef cooked a delicious meal.",
     "The chef cooked a delicious rock."),
    ("She drank a glass of cold water.",
     "She drank a glass of cold table."),
    ("He read an interesting book about history.",
     "He read an interesting lamp about history."),
    ("The farmer planted seeds in the soil.",
     "The farmer planted seeds in the ceiling."),
    ("The teacher gave the students a test.",
     "The teacher gave the students a cloud."),
    # Pronoun reference
    ("The boy found his lost dog and was happy.",
     "The boy found his lost dog and was purple."),
    ("Mary called her friend because she was lonely.",
     "Mary called her friend because she was wooden."),
    # World knowledge
    ("The sun rises in the east.",
     "The sun rises in the floor."),
    ("Fish live in water and breathe through gills.",
     "Fish live in trees and breathe through doors."),
    ("Birds have wings and can fly.",
     "Birds have wheels and can drive."),
    # Narrative coherence
    ("The boy was sad because his toy broke.",
     "The boy was sad because his toy flew to the moon and danced."),
    ("She smiled because she got a good grade.",
     "She smiled because the table ate her homework."),
    ("He was late because the bus didn't come on time.",
     "He was late because the color blue forgot to sing."),
    # Transitivity
    ("She kicked the ball across the field.",
     "She kicked the ball across the idea."),
    ("He threw the stone into the river.",
     "He threw the stone into the thought."),
    # Simple negation
    ("The room was not dark because the light was on.",
     "The room was not dark because the dark was dark."),
    ("She could not find her keys anywhere in the house.",
     "She could not find her keys anywhere in the nothing."),
    # Comparison
    ("The elephant is bigger than the mouse.",
     "The elephant is bigger than the bigger."),
    ("Summer is warmer than winter in most places.",
     "Summer is warmer than winter in most warmer."),
    # Spatial coherence
    ("The cup was on the table next to the plate.",
     "The cup was on the table next to the nothing."),
    ("The picture hung on the wall above the fireplace.",
     "The picture hung on the wall above the above."),
    # Common sense
    ("She wore a coat because it was cold outside.",
     "She wore a coat because it was coat outside."),
    ("He used an umbrella because it was raining.",
     "He used an umbrella because it was umbrella."),
    ("The ice cream melted because it was a hot day.",
     "The ice cream melted because it was a melted day."),
    # Functional
    ("He used a knife to cut the bread.",
     "He used a knife to drink the bread."),
    ("She used a broom to sweep the floor.",
     "She used a broom to eat the floor."),
    ("They used a map to find the way.",
     "They used a map to cook the way."),
    # Quantifier scope
    ("All the students passed the easy test.",
     "All the students passed the students test students."),
    ("Most children like to play outside.",
     "Most children like to play most."),
    ("Every morning she drinks a cup of tea.",
     "Every morning she drinks a cup of every."),
]


def _score_tier1(model: Evaluatable, max_texts: int) -> dict[str, float]:
    """Tier 1: Loss metrics on held-out text (reuses quick_loss data)."""
    tokenizer = model.tokenizer
    device = model.device
    texts = _HELD_OUT_TEXTS[:max_texts]

    total_loss = 0.0
    total_top1 = 0
    total_top5 = 0
    total_tokens = 0

    for text in texts:
        token_ids = tokenizer.encode(text)
        if len(token_ids) < 2:
            continue

        ll = model.loglikelihood_rolling(token_ids)
        n_scored = len(token_ids) - 1
        avg_nll = -ll / n_scored
        total_loss += avg_nll * n_scored

        ids_tensor = torch.tensor([token_ids], device=device)
        raw_model = getattr(model, '_model', None)
        if raw_model is not None:
            was_training = raw_model.training
            raw_model.eval()
            with torch.inference_mode():
                logits = raw_model(ids_tensor).logits
            if was_training:
                raw_model.train()

            pred_logits = logits[0, :-1, :]
            targets = torch.tensor(token_ids[1:], device=device)
            total_top1 += (pred_logits.argmax(dim=-1) == targets).sum().item()
            top5 = pred_logits.topk(min(5, pred_logits.size(-1)), dim=-1).indices
            total_top5 += (top5 == targets.unsqueeze(-1)).any(dim=-1).sum().item()
            total_tokens += n_scored

    if total_tokens == 0:
        return {"t1_loss": 5.0, "t1_ppl": 148.0, "t1_top1": 0.0, "t1_top5": 0.0}

    avg_loss = total_loss / total_tokens
    return {
        "t1_loss": round(avg_loss, 4),
        "t1_ppl": round(math.exp(min(avg_loss, 20)), 2),
        "t1_top1": round(total_top1 / total_tokens, 4),
        "t1_top5": round(total_top5 / total_tokens, 4),
    }


def _score_tier2(model: Evaluatable) -> dict[str, float]:
    """Tier 2: Phrase completion — is the expected word in top-K?"""
    tokenizer = model.tokenizer
    device = model.device
    raw_model = getattr(model, '_model', None)

    if raw_model is None:
        return {"t2_top1": 0.0, "t2_top5": 0.0, "t2_top10": 0.0}

    top1_correct = 0
    top5_correct = 0
    top10_correct = 0
    total = 0

    was_training = raw_model.training
    raw_model.eval()

    with torch.inference_mode():
        for context, expected in _PHRASE_COMPLETIONS:
            context_ids = tokenizer.encode(context)
            expected_ids = tokenizer.encode(expected)
            if not context_ids or not expected_ids:
                continue

            # We only check the first token of the expected word
            expected_first_token = expected_ids[0]

            ids_tensor = torch.tensor([context_ids], device=device)
            logits = raw_model(ids_tensor).logits
            # Last position predicts next token
            next_logits = logits[0, -1, :]  # (vocab_size,)

            top10_indices = next_logits.topk(min(10, next_logits.size(-1))).indices
            if expected_first_token == next_logits.argmax().item():
                top1_correct += 1
            if expected_first_token in top10_indices[:5].tolist():
                top5_correct += 1
            if expected_first_token in top10_indices.tolist():
                top10_correct += 1
            total += 1

    if was_training:
        raw_model.train()

    if total == 0:
        return {"t2_top1": 0.0, "t2_top5": 0.0, "t2_top10": 0.0}

    return {
        "t2_top1": round(top1_correct / total, 4),
        "t2_top5": round(top5_correct / total, 4),
        "t2_top10": round(top10_correct / total, 4),
    }


def _score_tier3(model: Evaluatable) -> dict[str, float]:
    """Tier 3: Coherence — prefer coherent vs incoherent sentences."""
    tokenizer = model.tokenizer
    correct = 0
    total = 0

    for good, bad in _COHERENCE_PAIRS:
        good_ids = tokenizer.encode(good)
        bad_ids = tokenizer.encode(bad)
        if not good_ids or not bad_ids:
            continue

        good_ll = model.loglikelihood_rolling(good_ids)
        bad_ll = model.loglikelihood_rolling(bad_ids)

        # Normalize by length to avoid length bias
        good_score = good_ll / len(good_ids)
        bad_score = bad_ll / len(bad_ids)

        if good_score > bad_score:
            correct += 1
        total += 1

    if total == 0:
        return {"t3_coherence": 0.0}

    return {"t3_coherence": round(correct / total, 4)}


def _score_tier4(model: Evaluatable, n_per_type: int, seed: int) -> dict[str, float]:
    """Tier 4: State tracking — counting, recall, pattern completion."""
    tokenizer = model.tokenizer

    type_correct: dict[str, list[float]] = {
        "counting": [], "recall": [], "pattern": [],
    }

    generators = [
        ("counting", _generate_counting_examples),
        ("recall", _generate_recall_examples),
        ("pattern", _generate_pattern_examples),
    ]

    for task_type, gen_fn in generators:
        examples = gen_fn(n_per_type, seed)
        for ex in examples:
            prompt_ids = tokenizer.encode(ex["prompt"])
            answer_ids = tokenizer.encode(ex["answer"])
            if not prompt_ids or not answer_ids:
                continue

            generated_ids = model.generate(
                prompt_ids,
                max_new_tokens=len(answer_ids),
                temperature=0.0,
            )
            predicted = tokenizer.decode(generated_ids).strip()
            target = ex["answer"].strip()
            type_correct[task_type].append(1.0 if predicted == target else 0.0)

    metrics: dict[str, float] = {}
    all_scores: list[float] = []
    for task_type, scores in type_correct.items():
        if scores:
            acc = sum(scores) / len(scores)
            metrics[f"t4_{task_type}"] = round(acc, 4)
            all_scores.extend(scores)

    metrics["t4_avg"] = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0
    return metrics


def _composite_score(metrics: dict[str, float]) -> float:
    """Compute weighted composite score in [0, 1]."""
    # Tier 1: normalize loss to [0, 1] — loss=5 → 0, loss=0 → 1
    t1 = max(0.0, min(1.0, 1.0 - metrics.get("t1_loss", 5.0) / 5.0))

    # Tier 2: top-5 phrase completion accuracy (already [0, 1])
    t2 = metrics.get("t2_top5", 0.0)

    # Tier 3: coherence accuracy (already [0, 1])
    t3 = metrics.get("t3_coherence", 0.0)

    # Tier 4: average state tracking accuracy (already [0, 1])
    t4 = metrics.get("t4_avg", 0.0)

    return round(0.3 * t1 + 0.2 * t2 + 0.3 * t3 + 0.2 * t4, 4)


@register_task("ablation_suite")
class AblationSuiteTask(EvalTask):
    """Fast multi-tier eval for ablation experiments."""

    name = "ablation_suite"

    def download(self, data_dir: str) -> None:
        pass  # All data is hardcoded

    def evaluate(
        self,
        model: Evaluatable,
        config: EvalConfig,
        on_progress: Optional[ProgressCallback] = None,
    ) -> EvalResult:
        max_texts = config.max_samples or 32
        n_state = min(20, config.max_samples or 20)
        seed = config.seed

        t0 = time.perf_counter()

        # Run all tiers
        metrics: dict[str, float] = {}

        if on_progress:
            on_progress(0, 4)

        t1_metrics = _score_tier1(model, max_texts)
        metrics.update(t1_metrics)
        if on_progress:
            on_progress(1, 4)

        t2_metrics = _score_tier2(model)
        metrics.update(t2_metrics)
        if on_progress:
            on_progress(2, 4)

        t3_metrics = _score_tier3(model)
        metrics.update(t3_metrics)
        if on_progress:
            on_progress(3, 4)

        t4_metrics = _score_tier4(model, n_state, seed)
        metrics.update(t4_metrics)
        if on_progress:
            on_progress(4, 4)

        # Composite
        metrics["composite"] = _composite_score(metrics)

        elapsed = time.perf_counter() - t0

        return EvalResult(
            task_name=self.name,
            metrics=metrics,
            metadata={
                "max_texts_t1": max_texts,
                "n_state_per_type": n_state,
                "elapsed_seconds": round(elapsed, 2),
            },
        )
