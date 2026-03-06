"""Evaluation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalConfig:
    """Configuration for running evaluations."""

    tasks: list[str] = field(default_factory=lambda: ["perplexity"])
    batch_size: int = 32
    max_samples: int | None = None  # None = use full dataset
    num_few_shot: int = 0
    seed: int = 42
    data_dir: str = "data/eval"  # cache directory for downloaded datasets
    interval: int = 2000  # run evals every N training steps


# ── Standard eval suites ─────────────────────────────────────────────────────
# Core tasks: perplexity (quality), blimp (grammar).

# Standard eval: the same benchmarks used by SmolLM / LLaMA evals.
STANDARD_EVAL = EvalConfig(
    tasks=["hellaswag", "arc_easy", "arc_challenge", "piqa", "winogrande", "mmlu"],
    max_samples=500,
    interval=2000,
)

# Quick eval: multi-tier ablation suite. Use for ablations / development.
QUICK_EVAL = EvalConfig(
    tasks=["ablation_suite"],
    max_samples=32,
    interval=500,
)

# All task names in the standard suite, for use in TrainingConfig defaults.
STANDARD_EVAL_TASKS = "hellaswag,arc_easy,arc_challenge,piqa,winogrande,mmlu"

# ── Eval preset registry ─────────────────────────────────────────────────────

EVAL_PRESETS: dict[str, EvalConfig] = {
    "standard": STANDARD_EVAL,
    "quick": QUICK_EVAL,
    "none": EvalConfig(tasks=[]),
}

EVAL_PRESET_LABELS: dict[str, str] = {
    "standard": "Standard (hellaswag, arc, piqa, winogrande, mmlu)",
    "quick": "Quick (ablation suite, 32 samples)",
    "none": "None (no eval)",
}


def get_eval_presets() -> list[dict[str, str]]:
    """Return list of {name, label} for all eval presets."""
    return [{"name": k, "label": EVAL_PRESET_LABELS[k]} for k in EVAL_PRESETS]


def get_eval_preset(name: str) -> EvalConfig | None:
    """Return an eval preset config by name, or None if not found."""
    return EVAL_PRESETS.get(name)
