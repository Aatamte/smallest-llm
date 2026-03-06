"""Named preset configurations for quick experiment setup."""

from __future__ import annotations

from src.config.base import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    StageConfig,
    TrainingConfig,
)

# ── Global FLOPs budgets ─────────────────────────────────────────────────────
# Compute-matched budgets for fair architecture comparison on MacBook Pro (MPS).
# Training stops when the cumulative FLOPs reach the budget.

FLOPS_QUICK = 1e14        # ~1 min   — fast iteration, architecture screening
FLOPS_STANDARD = 1e15     # ~8 min   — meaningful training, hyperparam search
FLOPS_FULL = 1e16         # ~1.5 hr  — serious training, benchmark comparisons
FLOPS_PRODUCTION = 5e16   # ~7 hr    — final models, push for best results

FLOPS_BUDGETS: dict[str, float] = {
    "quick": FLOPS_QUICK,
    "standard": FLOPS_STANDARD,
    "full": FLOPS_FULL,
    "production": FLOPS_PRODUCTION,
}

FLOPS_BUDGET_LABELS: dict[str, str] = {
    "quick": "Quick (1e14 FLOPs, ~1 min)",
    "standard": "Standard (1e15 FLOPs, ~8 min)",
    "full": "Full (1e16 FLOPs, ~1.5 hr)",
    "production": "Production (5e16 FLOPs, ~7 hr)",
}


def get_flops_budgets() -> list[dict[str, str]]:
    """Return list of {name, label} for FLOPs budget options."""
    return [{"name": k, "label": FLOPS_BUDGET_LABELS[k]} for k in FLOPS_BUDGETS]


def get_flops_budget(name: str) -> float | None:
    """Return FLOPs value for a named budget, or None."""
    return FLOPS_BUDGETS.get(name)

# ── Stage builders ───────────────────────────────────────────────────────────
# Stages split a FLOPs budget proportionally. Sanity stages stay step-based.


def make_full_stages(budget: float) -> list[StageConfig]:
    """Stages for ~2M models: short seqs, simple datasets."""
    return [
        StageConfig(
            name="sanity", max_steps=50, seq_len=32, dataset_name="tiny_stories",
            lr=1e-3, warmup_steps=0, overfit_batches=1, loss_threshold=0.5,
            eval_interval=25, log_interval=5, save_interval=50, eval_enabled=False,
        ),
        StageConfig(
            name="foundation", max_steps=0, max_flops=budget * 0.4,
            seq_len=64, dataset_name="tiny_stories", lr=1e-3, warmup_steps=100,
            eval_interval=100, log_interval=10, save_interval=500,
        ),
        StageConfig(
            name="extension", max_steps=0, max_flops=budget * 0.4,
            seq_len=128, dataset_name="minipile", lr=5e-4, warmup_steps=50,
            eval_interval=100, log_interval=10, save_interval=500,
        ),
        StageConfig(
            name="refinement", max_steps=0, max_flops=budget * 0.2,
            seq_len=256, dataset_name="openwebtext", lr=1e-4, warmup_steps=25,
            min_lr_ratio=0.01, eval_interval=100, log_interval=10, save_interval=500,
        ),
    ]


def make_20m_stages(budget: float) -> list[StageConfig]:
    """Stages for ~20M models: longer seqs, higher LRs."""
    return [
        StageConfig(
            name="sanity", max_steps=50, seq_len=64, dataset_name="tiny_stories",
            lr=5e-3, warmup_steps=0, overfit_batches=1, loss_threshold=0.5,
            eval_interval=25, log_interval=5, save_interval=50, eval_enabled=False,
        ),
        StageConfig(
            name="foundation", max_steps=0, max_flops=budget * 0.4,
            seq_len=256, dataset_name="tiny_stories", lr=5e-3, warmup_steps=200,
            eval_interval=500, log_interval=10, save_interval=2000,
        ),
        StageConfig(
            name="extension", max_steps=0, max_flops=budget * 0.4,
            seq_len=512, dataset_name="minipile", lr=3e-3, warmup_steps=100,
            eval_interval=500, log_interval=10, save_interval=2000,
        ),
        StageConfig(
            name="refinement", max_steps=0, max_flops=budget * 0.2,
            seq_len=512, dataset_name="openwebtext", lr=1e-3, warmup_steps=50,
            min_lr_ratio=0.01, eval_interval=500, log_interval=10, save_interval=2000,
        ),
    ]


def make_production_stages(budget: float) -> list[StageConfig]:
    """Stages for ~30M production models: long seqs, careful LR schedule."""
    return [
        StageConfig(
            name="sanity", max_steps=50, seq_len=64, dataset_name="tiny_stories",
            lr=3e-3, warmup_steps=0, overfit_batches=1, loss_threshold=0.5,
            eval_interval=25, log_interval=5, save_interval=50, eval_enabled=False,
        ),
        StageConfig(
            name="foundation", max_steps=0, max_flops=budget * 0.4,
            seq_len=256, dataset_name="tiny_stories", lr=3e-3, warmup_steps=500,
            eval_interval=1000, log_interval=10, save_interval=5000,
        ),
        StageConfig(
            name="extension", max_steps=0, max_flops=budget * 0.4,
            seq_len=512, dataset_name="minipile", lr=1.5e-3, warmup_steps=200,
            eval_interval=1000, log_interval=10, save_interval=5000,
        ),
        StageConfig(
            name="refinement", max_steps=0, max_flops=budget * 0.2,
            seq_len=512, dataset_name="openwebtext", lr=5e-4, warmup_steps=100,
            min_lr_ratio=0.01, eval_interval=1000, log_interval=10, save_interval=5000,
        ),
    ]


# ── Model configs ─────────────────────────────────────────────────────────────

TRANSFORMER_MODEL = ModelConfig(
    name="transformer",
    extra_args={"d_model": 128, "n_heads": 4, "n_layers": 4},
)
MAMBA_MODEL = ModelConfig(
    name="mamba",
    extra_args={"d_model": 128, "n_layers": 7, "d_state": 16, "d_conv": 4, "expand_factor": 2},
)
MAMBA2_MODEL = ModelConfig(
    name="mamba2",
    extra_args={"d_model": 128, "n_layers": 7, "d_state": 16, "d_conv": 4, "expand_factor": 2, "chunk_size": 64},
)
MAMBA3_MODEL = ModelConfig(
    name="mamba3",
    extra_args={"d_model": 128, "n_layers": 7, "d_state": 16, "expand_factor": 2, "chunk_size": 64, "mlp_factor": 4},
)
IMPROVED_MAMBA3_MODEL = ModelConfig(
    name="improved_mamba3",
    extra_args={"d_model": 128, "n_layers": 7, "d_state": 16, "expand_factor": 2, "chunk_size": 64, "mlp_factor": 4},
)
IMPROVED_MAMBA3_20M_MODEL = ModelConfig(
    name="improved_mamba3",
    extra_args={
        "d_model": 256, "n_layers": 16, "d_state": 32,
        "expand_factor": 2, "chunk_size": 64, "mlp_factor": 4,
        "gradient_checkpointing": True,
    },
)

TRANSFORMER_10M_MODEL = ModelConfig(
    name="transformer",
    extra_args={"d_model": 384, "n_heads": 6, "n_layers": 6, "dropout": 0.1},
)

MOE_2M_MODEL = ModelConfig(
    name="moe_transformer",
    extra_args={
        "d_model": 128, "n_heads": 4, "n_kv_heads": 2, "n_layers": 8,
        "mlp_ratio": 2.67, "n_experts": 4, "top_k": 1, "aux_loss_weight": 0.01,
    },
)

MODERN_2M_MODEL = ModelConfig(
    name="modern_transformer",
    extra_args={"d_model": 128, "n_heads": 4, "n_kv_heads": 2, "n_layers": 11, "mlp_ratio": 2.67},
)

MODERN_10M_MODEL = ModelConfig(
    name="modern_transformer",
    extra_args={"d_model": 256, "n_heads": 8, "n_kv_heads": 4, "n_layers": 12, "mlp_ratio": 2.67},
)

PRODUCTION_MODEL = ModelConfig(
    name="improved_mamba3",
    extra_args={
        "d_model": 320, "n_layers": 16, "d_state": 32,
        "expand_factor": 2, "chunk_size": 64, "mlp_factor": 4,
        "gradient_checkpointing": True,
    },
)

# ── Data configs ──────────────────────────────────────────────────────────────

QUICK_DATA = DataConfig(
    dataset_name="tiny_stories",
    tokenizer_name="byte",
    max_seq_len=64,
    batch_size=32,
    max_eval_batches=50,
)
FULL_DATA = DataConfig(
    dataset_name="tiny_stories",
    tokenizer_name="byte",
    max_seq_len=256,
    batch_size=32,
    max_eval_batches=50,
)
PRODUCTION_DATA = DataConfig(
    dataset_name="tiny_stories",
    tokenizer_name="byte",
    max_seq_len=512,
    batch_size=12,
    max_eval_batches=50,
    num_workers=2,
)
DATA_20M = DataConfig(
    dataset_name="tiny_stories",
    tokenizer_name="byte",
    max_seq_len=512,
    batch_size=16,
    max_eval_batches=50,
    num_workers=2,
)

# ── Optimizer configs ─────────────────────────────────────────────────────────

ADAMW_OPTIMIZER = OptimizerConfig(lr=1e-3)
MUON_OPTIMIZER = OptimizerConfig(name="muon", lr=0.02, beta1=0.95, grad_clip_norm=0.0)
MUON_20M_OPTIMIZER = OptimizerConfig(name="muon", lr=0.005, beta1=0.95, grad_clip_norm=0.0)
MUON_PRODUCTION_OPTIMIZER = OptimizerConfig(name="muon", lr=0.003, beta1=0.95, grad_clip_norm=0.0)

# ── Training configs (FLOPs-based) ───────────────────────────────────────────
# Default budgets per model size tier. Overridden by the FLOPs budget selector.

TRAINING_SMALL = TrainingConfig(
    max_steps=0, max_flops=FLOPS_STANDARD,
    eval_interval=100, log_interval=10, save_interval=500,
    eval_loss=False, multi_token=True, multi_token_n_ahead=4,
)
TRAINING_MEDIUM = TrainingConfig(
    max_steps=0, max_flops=FLOPS_FULL,
    eval_interval=500, log_interval=10, save_interval=2000,
    mixed_precision=True, eval_loss=False, multi_token=True, multi_token_n_ahead=4,
)
TRAINING_LARGE = TrainingConfig(
    max_steps=0, max_flops=FLOPS_PRODUCTION,
    eval_interval=1000, log_interval=10, save_interval=5000,
    mixed_precision=True, compile_model=True, eval_loss=False, multi_token=True, multi_token_n_ahead=4,
)

# ── Presets ───────────────────────────────────────────────────────────────────
# One preset per architecture. FLOPs budget is selected separately in the UI.

PRESETS: dict[str, ExperimentConfig] = {
    "transformer": ExperimentConfig(
        name="transformer",
        model=TRANSFORMER_MODEL,
        data=FULL_DATA,
        training=TrainingConfig(
            max_steps=0, max_flops=FLOPS_STANDARD,
            eval_interval=100, log_interval=10, save_interval=500,
            multi_token=True, multi_token_n_ahead=4, eval_loss=False,
        ),
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=100),
        stages=make_full_stages(FLOPS_STANDARD),
    ),
    "mamba": ExperimentConfig(
        name="mamba",
        model=MAMBA_MODEL,
        data=FULL_DATA,
        training=TRAINING_SMALL,
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=100),
        stages=make_full_stages(FLOPS_STANDARD),
    ),
    "mamba2": ExperimentConfig(
        name="mamba2",
        model=MAMBA2_MODEL,
        data=FULL_DATA,
        training=TRAINING_SMALL,
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=100),
        stages=make_full_stages(FLOPS_STANDARD),
    ),
    "mamba3": ExperimentConfig(
        name="mamba3",
        model=MAMBA3_MODEL,
        data=FULL_DATA,
        training=TRAINING_SMALL,
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=100),
        stages=make_full_stages(FLOPS_STANDARD),
    ),
    "improved-mamba3": ExperimentConfig(
        name="improved-mamba3",
        model=IMPROVED_MAMBA3_MODEL,
        data=FULL_DATA,
        training=TRAINING_SMALL,
        optimizer=MUON_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=100),
        stages=make_full_stages(FLOPS_STANDARD),
    ),
    "improved-mamba3-20m": ExperimentConfig(
        name="improved-mamba3-20m",
        model=IMPROVED_MAMBA3_20M_MODEL,
        data=DATA_20M,
        training=TRAINING_MEDIUM,
        optimizer=MUON_20M_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=200),
        stages=make_20m_stages(FLOPS_FULL),
    ),
    "production-30m": ExperimentConfig(
        name="production-30m",
        model=PRODUCTION_MODEL,
        data=PRODUCTION_DATA,
        training=TRAINING_LARGE,
        optimizer=MUON_PRODUCTION_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=500),
        stages=make_production_stages(FLOPS_PRODUCTION),
    ),
    "transformer-10m": ExperimentConfig(
        name="transformer-10m",
        model=TRANSFORMER_10M_MODEL,
        data=DATA_20M,
        training=TrainingConfig(
            max_steps=0, max_flops=FLOPS_FULL,
            eval_interval=500, log_interval=10, save_interval=2000,
            multi_token=True, multi_token_n_ahead=4, eval_loss=False,
        ),
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=200),
        stages=make_20m_stages(FLOPS_FULL),
    ),
    "moe-2m": ExperimentConfig(
        name="moe-2m",
        model=MOE_2M_MODEL,
        data=DataConfig(dataset_name="tiny_stories", tokenizer_name="byte", max_seq_len=256, batch_size=32, max_eval_batches=50, num_workers=2),
        training=TrainingConfig(
            max_steps=0, max_flops=FLOPS_STANDARD,
            eval_interval=100, log_interval=10, save_interval=500,
            mixed_precision=True, multi_token=True, multi_token_n_ahead=4, eval_loss=False,
        ),
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=100),
        stages=make_full_stages(FLOPS_STANDARD),
    ),
    "modern-2m": ExperimentConfig(
        name="modern-2m",
        model=MODERN_2M_MODEL,
        data=DataConfig(dataset_name="tiny_stories", tokenizer_name="byte", max_seq_len=256, batch_size=32, max_eval_batches=50, num_workers=2),
        training=TrainingConfig(
            max_steps=0, max_flops=FLOPS_STANDARD,
            eval_interval=100, log_interval=10, save_interval=500,
            mixed_precision=True, multi_token=True, multi_token_n_ahead=4, eval_loss=False,
        ),
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=100),
        stages=make_full_stages(FLOPS_STANDARD),
    ),
    "modern-10m": ExperimentConfig(
        name="modern-10m",
        model=MODERN_10M_MODEL,
        data=DataConfig(dataset_name="tiny_stories", tokenizer_name="byte", max_seq_len=512, batch_size=16, max_eval_batches=50, num_workers=2),
        training=TrainingConfig(
            max_steps=0, max_flops=FLOPS_FULL,
            eval_interval=500, log_interval=10, save_interval=2000,
            mixed_precision=True, multi_token=True, multi_token_n_ahead=4, eval_loss=False,
        ),
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=200),
        stages=make_20m_stages(FLOPS_FULL),
    ),
    "sanity-check": ExperimentConfig(
        name="sanity-check",
        model=TRANSFORMER_MODEL,
        data=DataConfig(dataset_name="tiny_stories", tokenizer_name="byte", max_seq_len=32, batch_size=32, max_eval_batches=50),
        training=TrainingConfig(max_steps=100, eval_interval=10, log_interval=5, save_interval=100, eval_loss=False, multi_token=True, multi_token_n_ahead=4),
        optimizer=ADAMW_OPTIMIZER,
        stages=[
            StageConfig(
                name="sanity",
                max_steps=100,
                seq_len=32,
                dataset_name="tiny_stories",
                lr=1e-3,
                warmup_steps=0,
                overfit_batches=1,
                loss_threshold=0.5,
                eval_interval=50,
                log_interval=5,
                save_interval=100,
                eval_enabled=False,
            ),
        ],
    ),
}

# Human-readable labels — architecture + params only, no FLOPs
PRESET_LABELS: dict[str, str] = {
    "transformer": "Transformer (0.86M, multi-token)",
    "mamba": "Mamba (0.85M)",
    "mamba2": "Mamba-2 (0.85M)",
    "mamba3": "Mamba-3 (2.14M)",
    "improved-mamba3": "Improved Mamba-3 (2.14M, Muon)",
    "improved-mamba3-20m": "Improved Mamba-3 20M (19.4M, Muon)",
    "production-30m": "Production Mamba-3 (~30M, Muon)",
    "transformer-10m": "Transformer 10M (~10M, multi-token)",
    "moe-2m": "MoE Transformer (~5M/~1.5M active, 4 experts)",
    "modern-2m": "Modern Transformer 2M (RoPE+GQA+SwiGLU)",
    "modern-10m": "Modern Transformer 10M (8.8M, RoPE+GQA+SwiGLU)",
    "sanity-check": "Sanity Check (overfit 1 batch)",
}


def describe_preset(config: ExperimentConfig) -> str:
    """Auto-compute a human-readable description from a preset's config."""
    stages = config.stages
    if not stages:
        return f"on {config.data.dataset_name}"

    stage_names = [s.name for s in stages if s.overfit_batches == 0]
    return " → ".join(stage_names)


def get_presets() -> list[dict[str, str]]:
    """Return list of {name, label, description} for all presets."""
    return [
        {"name": k, "label": PRESET_LABELS.get(k, k), "description": describe_preset(v)}
        for k, v in PRESETS.items()
    ]


def get_preset(name: str) -> ExperimentConfig | None:
    """Return a preset config by name, or None if not found."""
    return PRESETS.get(name)
