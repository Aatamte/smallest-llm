"""BLiMP (Benchmark of Linguistic Minimal Pairs) evaluation.

67 sub-datasets, each with 1000 minimal pairs testing whether the model
assigns higher probability to grammatically correct sentences.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

import requests

from src.evaluation.config import EvalConfig
from src.evaluation.protocol import Evaluatable
from src.evaluation.results import EvalResult
from src.evaluation.tasks import register_task
from src.evaluation.tasks.base import EvalTask

_BLIMP_BASE_URL = (
    "https://raw.githubusercontent.com/alexwarstadt/blimp/master/data/"
)

# BLiMP sub-datasets grouped by linguistic field.
# Categories follow the original BLiMP paper (Warstadt et al., 2020).
_BLIMP_SUBTASKS: dict[str, list[str]] = {
    "morphology": [
        "anaphor_gender_agreement",
        "anaphor_number_agreement",
        "determiner_noun_agreement_1",
        "determiner_noun_agreement_2",
        "determiner_noun_agreement_irregular_1",
        "determiner_noun_agreement_irregular_2",
        "determiner_noun_agreement_with_adj_2",
        "determiner_noun_agreement_with_adj_irregular_1",
        "determiner_noun_agreement_with_adj_irregular_2",
        "determiner_noun_agreement_with_adjective_1",
        "distractor_agreement_relational_noun",
        "distractor_agreement_relative_clause",
        "irregular_past_participle_adjectives",
        "irregular_past_participle_verbs",
        "irregular_plural_subject_verb_agreement_1",
        "irregular_plural_subject_verb_agreement_2",
        "regular_plural_subject_verb_agreement_1",
        "regular_plural_subject_verb_agreement_2",
    ],
    "syntax": [
        "adjunct_island",
        "animate_subject_passive",
        "animate_subject_trans",
        "causative",
        "complex_NP_island",
        "coordinate_structure_constraint_complex_left_branch",
        "coordinate_structure_constraint_object_extraction",
        "drop_argument",
        "ellipsis_n_bar_1",
        "ellipsis_n_bar_2",
        "inchoative",
        "intransitive",
        "left_branch_island_echo_question",
        "left_branch_island_simple_question",
        "passive_1",
        "passive_2",
        "sentential_subject_island",
        "transitive",
        "wh_island",
        "wh_questions_object_gap",
        "wh_questions_subject_gap",
        "wh_questions_subject_gap_long_distance",
        "wh_vs_that_no_gap",
        "wh_vs_that_no_gap_long_distance",
        "wh_vs_that_with_gap",
        "wh_vs_that_with_gap_long_distance",
    ],
    "syntax_semantics": [
        "existential_there_object_raising",
        "existential_there_quantifiers_1",
        "existential_there_quantifiers_2",
        "existential_there_subject_raising",
        "expletive_it_object_raising",
        "principle_A_c_command",
        "principle_A_case_1",
        "principle_A_case_2",
        "principle_A_domain_1",
        "principle_A_domain_2",
        "principle_A_domain_3",
        "principle_A_reconstruction",
        "tough_vs_raising_1",
        "tough_vs_raising_2",
    ],
    "semantics": [
        "matrix_question_npi_licensor_present",
        "npi_present_1",
        "npi_present_2",
        "only_npi_licensor_present",
        "only_npi_scope",
        "sentential_negation_npi_licensor_present",
        "sentential_negation_npi_scope",
        "superlative_quantifiers_1",
        "superlative_quantifiers_2",
    ],
}

# Flat list of all subtask names
_ALL_SUBTASKS: list[str] = []
for _subtasks in _BLIMP_SUBTASKS.values():
    _ALL_SUBTASKS.extend(_subtasks)


@register_task("blimp")
class BLiMPTask(EvalTask):
    """Evaluate linguistic knowledge via minimal pairs."""

    name = "blimp"

    def download(self, data_dir: str) -> None:
        dest_dir = Path(data_dir) / "blimp"
        dest_dir.mkdir(parents=True, exist_ok=True)

        for subtask in _ALL_SUBTASKS:
            dest = dest_dir / f"{subtask}.jsonl"
            if dest.exists():
                continue
            url = f"{_BLIMP_BASE_URL}{subtask}.jsonl"
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            dest.write_text(resp.text, encoding="utf-8")

    def evaluate(self, model: Evaluatable, config: EvalConfig) -> EvalResult:
        self.download(config.data_dir)
        blimp_dir = Path(config.data_dir) / "blimp"
        tokenizer = model.tokenizer

        t0 = time.perf_counter()

        subtask_scores: dict[str, list[float]] = defaultdict(list)
        field_scores: dict[str, list[float]] = defaultdict(list)
        all_scores: list[float] = []
        per_sample: list[dict] = []

        # Map subtask -> field
        subtask_to_field = {}
        for field_name, subtasks in _BLIMP_SUBTASKS.items():
            for st in subtasks:
                subtask_to_field[st] = field_name

        for subtask in _ALL_SUBTASKS:
            path = blimp_dir / f"{subtask}.jsonl"
            if not path.exists():
                continue

            pairs = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        pairs.append(json.loads(line))

            if config.max_samples is not None:
                pairs = pairs[: config.max_samples]

            for pair in pairs:
                good_text = pair["sentence_good"]
                bad_text = pair["sentence_bad"]

                good_ids = tokenizer.encode(good_text)
                bad_ids = tokenizer.encode(bad_text)

                if not good_ids or not bad_ids:
                    continue

                good_ll = model.loglikelihood_rolling(good_ids)
                bad_ll = model.loglikelihood_rolling(bad_ids)

                correct = 1.0 if good_ll > bad_ll else 0.0

                subtask_scores[subtask].append(correct)
                field_name = subtask_to_field.get(subtask, "other")
                field_scores[field_name].append(correct)
                all_scores.append(correct)

                if config.max_samples is not None and config.max_samples <= 100:
                    per_sample.append({
                        "subtask": subtask,
                        "good": good_text,
                        "bad": bad_text,
                        "good_ll": round(good_ll, 4),
                        "bad_ll": round(bad_ll, 4),
                        "correct": bool(correct),
                    })

        elapsed = time.perf_counter() - t0

        # Compute metrics
        overall_acc = _mean(all_scores)
        metrics: dict[str, float] = {"accuracy": round(overall_acc, 4)}

        for field_name, scores in sorted(field_scores.items()):
            metrics[f"accuracy_{field_name}"] = round(_mean(scores), 4)

        # Per-subtask breakdown in metadata
        subtask_accs = {}
        for subtask, scores in sorted(subtask_scores.items()):
            subtask_accs[subtask] = round(_mean(scores), 4)

        return EvalResult(
            task_name=self.name,
            metrics=metrics,
            metadata={
                "num_pairs": len(all_scores),
                "num_subtasks": len(subtask_scores),
                "elapsed_seconds": round(elapsed, 2),
                "subtask_accuracy": subtask_accs,
            },
            per_sample=per_sample if per_sample else None,
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
