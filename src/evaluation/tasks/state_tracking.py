"""State tracking evaluation — synthetic probes for SSM capabilities.

Tests whether the model's recurrent state can track information:
- Counting: how many times does a character appear?
- Recall: remember a name/word from earlier in context
- Pattern completion: continue a repeating pattern
- Bracket matching: is a bracket sequence balanced?

These are synthetic tasks with known ground truth. No downloads needed.
"""

from __future__ import annotations

import random
import time
from typing import Optional

from src.evaluation.config import EvalConfig
from src.evaluation.protocol import Evaluatable
from src.evaluation.results import EvalResult
from src.evaluation.tasks import register_task
from src.evaluation.tasks.base import EvalTask, ProgressCallback


def _generate_counting_examples(n: int, seed: int = 42) -> list[dict]:
    """Generate counting task examples.

    Format: "aabba" -> model must predict the count of 'a' (3).
    We frame it as: "aabba. Count of a: " -> model should assign
    highest probability to the correct digit.
    """
    rng = random.Random(seed)
    examples = []
    chars = "abcde"
    for _ in range(n):
        length = rng.randint(3, 12)
        seq = "".join(rng.choice(chars) for _ in range(length))
        target_char = rng.choice(chars)
        count = seq.count(target_char)
        # Cap at 9 for single-digit answers
        if count > 9:
            continue
        examples.append({
            "prompt": f"{seq}. Count of {target_char}: ",
            "answer": str(count),
            "type": "counting",
        })
    return examples


def _generate_recall_examples(n: int, seed: int = 42) -> list[dict]:
    """Generate recall task examples.

    Format: "The cat is named Luna. The cat likes fish. The cat's name is "
    -> model should predict "Luna"
    """
    rng = random.Random(seed)
    names = ["Alice", "Bob", "Luna", "Max", "Sam", "Rose", "Jack", "Lily"]
    animals = ["cat", "dog", "bird", "fish", "fox"]
    actions = ["likes to run", "likes to sleep", "is very small", "is very old", "is happy"]
    examples = []
    for _ in range(n):
        name = rng.choice(names)
        animal = rng.choice(animals)
        # Add 1-3 distractor sentences
        n_distractors = rng.randint(1, 3)
        sentences = [f"The {animal} is named {name}."]
        for _ in range(n_distractors):
            sentences.append(f"The {animal} {rng.choice(actions)}.")
        sentences.append(f"The {animal}'s name is ")
        prompt = " ".join(sentences)
        examples.append({
            "prompt": prompt,
            "answer": name,
            "type": "recall",
        })
    return examples


def _generate_pattern_examples(n: int, seed: int = 42) -> list[dict]:
    """Generate pattern completion examples.

    Format: "ABCABCABC" -> model should predict "A"
    """
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        # Random pattern of length 2-5
        pat_len = rng.randint(2, 5)
        chars = [chr(65 + i) for i in range(pat_len)]  # A, B, C, ...
        pattern = "".join(chars)
        # Repeat 2-4 times
        reps = rng.randint(2, 4)
        seq = pattern * reps
        # The next char should be the first char of the pattern
        examples.append({
            "prompt": seq,
            "answer": chars[0],
            "type": "pattern",
        })
    return examples


def _generate_bracket_examples(n: int, seed: int = 42) -> list[dict]:
    """Generate bracket matching examples.

    Format: "(())" -> "balanced" vs "(()" -> "unbalanced"
    Framed as: "Is (()) balanced? " -> model should prefer "yes"/"no"
    """
    rng = random.Random(seed)
    examples = []
    for _ in range(n):
        length = rng.randint(2, 8) * 2  # even length
        # Generate balanced or unbalanced
        balanced = rng.random() < 0.5
        if balanced:
            # Generate a balanced sequence
            seq = _random_balanced_brackets(length, rng)
        else:
            # Generate unbalanced by random shuffling
            seq = "".join(rng.choice("()") for _ in range(length))
            # Make sure it's actually unbalanced
            if _is_balanced(seq):
                # Flip one bracket
                idx = rng.randint(0, len(seq) - 1)
                seq = seq[:idx] + (")" if seq[idx] == "(" else "(") + seq[idx + 1:]

        actual_balanced = _is_balanced(seq)
        answer = "yes" if actual_balanced else "no"
        examples.append({
            "prompt": f"Is {seq} balanced? ",
            "answer": answer,
            "type": "brackets",
        })
    return examples


def _random_balanced_brackets(length: int, rng: random.Random) -> str:
    """Generate a random balanced bracket sequence."""
    if length <= 0:
        return ""
    # Simple approach: generate valid sequence
    seq = []
    open_count = 0
    for i in range(length):
        remaining = length - i
        if open_count == 0:
            seq.append("(")
            open_count += 1
        elif open_count == remaining:
            seq.append(")")
            open_count -= 1
        elif rng.random() < 0.5:
            seq.append("(")
            open_count += 1
        else:
            seq.append(")")
            open_count -= 1
    return "".join(seq)


def _is_balanced(s: str) -> bool:
    depth = 0
    for c in s:
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        if depth < 0:
            return False
    return depth == 0


@register_task("state_tracking")
class StateTrackingTask(EvalTask):
    """Evaluate SSM state tracking via synthetic probes."""

    name = "state_tracking"

    def download(self, data_dir: str) -> None:
        pass  # All data is generated synthetically

    def evaluate(
        self,
        model: Evaluatable,
        config: EvalConfig,
        on_progress: Optional[ProgressCallback] = None,
    ) -> EvalResult:
        seed = config.seed
        n_per_type = config.max_samples or 100

        # Generate all examples
        examples = []
        examples.extend(_generate_counting_examples(n_per_type, seed))
        examples.extend(_generate_recall_examples(n_per_type, seed))
        examples.extend(_generate_pattern_examples(n_per_type, seed))
        examples.extend(_generate_bracket_examples(n_per_type, seed))

        tokenizer = model.tokenizer
        total = len(examples)
        if on_progress:
            on_progress(0, total)

        t0 = time.perf_counter()

        # Score each example
        type_correct: dict[str, list[float]] = {
            "counting": [], "recall": [], "pattern": [], "brackets": [],
        }

        per_sample: list[dict] = []

        for i, ex in enumerate(examples):
            prompt_ids = tokenizer.encode(ex["prompt"])
            answer_ids = tokenizer.encode(ex["answer"])

            if not prompt_ids or not answer_ids:
                continue

            # Check if greedy generation matches the answer
            generated_ids = model.generate(
                prompt_ids,
                max_new_tokens=len(answer_ids),
                temperature=0.0,
            )
            predicted = tokenizer.decode(generated_ids).strip()
            target = ex["answer"].strip()
            correct = 1.0 if predicted == target else 0.0

            type_correct[ex["type"]].append(correct)

            if n_per_type <= 50:
                per_sample.append({
                    "type": ex["type"],
                    "prompt": ex["prompt"][:80],
                    "target": target,
                    "predicted": predicted,
                    "correct": bool(correct),
                })

            if on_progress:
                on_progress(i + 1, total)

        elapsed = time.perf_counter() - t0

        # Compute metrics
        metrics: dict[str, float] = {}
        all_scores: list[float] = []
        for task_type, scores in type_correct.items():
            if scores:
                acc = sum(scores) / len(scores)
                metrics[f"acc_{task_type}"] = round(acc, 4)
                all_scores.extend(scores)

        if all_scores:
            metrics["accuracy"] = round(sum(all_scores) / len(all_scores), 4)
        else:
            metrics["accuracy"] = 0.0

        return EvalResult(
            task_name=self.name,
            metrics=metrics,
            metadata={
                "num_examples": len(all_scores),
                "n_per_type": n_per_type,
                "elapsed_seconds": round(elapsed, 2),
            },
            per_sample=per_sample if per_sample else None,
        )
