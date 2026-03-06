"""Named preset configurations for quick experiment setup."""

from __future__ import annotations

from copy import deepcopy

from src.config.base import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    StageConfig,
    TrainingConfig,
)

# ── Shared stage definitions ─────────────────────────────────────────────────

PRODUCTION_STAGES = [
    StageConfig(
        name="sanity",
        max_steps=50,
        seq_len=64,
        dataset_name="tiny_stories",
        lr=3e-3,
        warmup_steps=0,
        overfit_batches=1,
        loss_threshold=0.5,
        eval_interval=25,
        log_interval=5,
        save_interval=50,
        eval_enabled=False,
    ),
    StageConfig(
        name="foundation",
        max_steps=20000,
        seq_len=256,
        dataset_name="tiny_stories",
        lr=3e-3,
        warmup_steps=500,
        eval_interval=1000,
        log_interval=10,
        save_interval=5000,
    ),
    StageConfig(
        name="extension",
        max_steps=20000,
        seq_len=512,
        dataset_name="minipile",
        lr=1.5e-3,
        warmup_steps=200,
        eval_interval=1000,
        log_interval=10,
        save_interval=5000,
    ),
    StageConfig(
        name="refinement",
        max_steps=10000,
        seq_len=512,
        dataset_name="openwebtext",
        lr=5e-4,
        warmup_steps=100,
        min_lr_ratio=0.01,
        eval_interval=1000,
        log_interval=10,
        save_interval=5000,
    ),
]

STAGES_20M = [
    StageConfig(
        name="sanity",
        max_steps=50,
        seq_len=64,
        dataset_name="tiny_stories",
        lr=5e-3,
        warmup_steps=0,
        overfit_batches=1,
        loss_threshold=0.5,
        eval_interval=25,
        log_interval=5,
        save_interval=50,
        eval_enabled=False,
    ),
    StageConfig(
        name="foundation",
        max_steps=8000,
        seq_len=256,
        dataset_name="tiny_stories",
        lr=5e-3,
        warmup_steps=200,
        eval_interval=500,
        log_interval=10,
        save_interval=2000,
    ),
    StageConfig(
        name="extension",
        max_steps=8000,
        seq_len=512,
        dataset_name="minipile",
        lr=3e-3,
        warmup_steps=100,
        eval_interval=500,
        log_interval=10,
        save_interval=2000,
    ),
    StageConfig(
        name="refinement",
        max_steps=4000,
        seq_len=512,
        dataset_name="openwebtext",
        lr=1e-3,
        warmup_steps=50,
        min_lr_ratio=0.01,
        eval_interval=500,
        log_interval=10,
        save_interval=2000,
    ),
]

FULL_STAGES = [
    StageConfig(
        name="sanity",
        max_steps=50,
        seq_len=32,
        dataset_name="tiny_stories",
        lr=1e-3,
        warmup_steps=0,
        overfit_batches=1,
        loss_threshold=0.5,
        eval_interval=25,
        log_interval=5,
        save_interval=50,
        eval_enabled=False,
    ),
    StageConfig(
        name="foundation",
        max_steps=2000,
        seq_len=64,
        dataset_name="tiny_stories",
        lr=1e-3,
        warmup_steps=100,
        eval_interval=100,
        log_interval=10,
        save_interval=500,
    ),
    StageConfig(
        name="extension",
        max_steps=2000,
        seq_len=128,
        dataset_name="minipile",
        lr=5e-4,
        warmup_steps=50,
        eval_interval=100,
        log_interval=10,
        save_interval=500,
    ),
    StageConfig(
        name="refinement",
        max_steps=1000,
        seq_len=256,
        dataset_name="openwebtext",
        lr=1e-4,
        warmup_steps=25,
        min_lr_ratio=0.01,
        eval_interval=100,
        log_interval=10,
        save_interval=500,
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

# ── Training configs ──────────────────────────────────────────────────────────

QUICK_TRAINING = TrainingConfig(
    max_steps=200,
    eval_interval=50,
    log_interval=5,
    save_interval=100,
)
FULL_TRAINING = TrainingConfig(
    max_steps=5050,
    eval_interval=100,
    log_interval=10,
    save_interval=500,
)
PRODUCTION_TRAINING = TrainingConfig(
    max_steps=50_000,
    eval_interval=1000,
    log_interval=10,
    save_interval=5000,
    mixed_precision=True,
    compile_model=True,
)
TRAINING_20M = TrainingConfig(
    max_steps=20_000,
    eval_interval=500,
    log_interval=10,
    save_interval=2000,
    mixed_precision=True,
)

# ── Helper ────────────────────────────────────────────────────────────────────


def _quick(name: str, model: ModelConfig, optimizer: OptimizerConfig = ADAMW_OPTIMIZER) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        model=model,
        data=QUICK_DATA,
        training=QUICK_TRAINING,
        optimizer=optimizer,
        scheduler=SchedulerConfig(warmup_steps=20),
    )


def _full(name: str, model: ModelConfig, optimizer: OptimizerConfig = ADAMW_OPTIMIZER) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        model=model,
        data=FULL_DATA,
        training=FULL_TRAINING,
        optimizer=optimizer,
        scheduler=SchedulerConfig(warmup_steps=100),
        stages=deepcopy(FULL_STAGES),
    )


# ── Presets ───────────────────────────────────────────────────────────────────

PRESETS: dict[str, ExperimentConfig] = {
    # Transformer
    "quick-transformer": _quick("quick-transformer", TRANSFORMER_MODEL),
    "full-transformer": ExperimentConfig(
        name="full-transformer",
        model=TRANSFORMER_MODEL,
        data=FULL_DATA,
        training=TrainingConfig(
            max_steps=5050,
            eval_interval=100,
            log_interval=10,
            save_interval=500,
            multi_token=True,
            multi_token_n_ahead=4,
        ),
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=100),
        stages=deepcopy(FULL_STAGES),
    ),
    # Mamba
    "quick-mamba": _quick("quick-mamba", MAMBA_MODEL),
    "full-mamba": _full("full-mamba", MAMBA_MODEL),
    # Mamba-2
    "quick-mamba2": _quick("quick-mamba2", MAMBA2_MODEL),
    "full-mamba2": _full("full-mamba2", MAMBA2_MODEL),
    # Mamba-3
    "quick-mamba3": _quick("quick-mamba3", MAMBA3_MODEL),
    "full-mamba3": _full("full-mamba3", MAMBA3_MODEL),
    # Improved Mamba-3 (uses Muon optimizer)
    "quick-improved-mamba3": _quick("quick-improved-mamba3", IMPROVED_MAMBA3_MODEL, MUON_OPTIMIZER),
    "full-improved-mamba3": _full("full-improved-mamba3", IMPROVED_MAMBA3_MODEL, MUON_OPTIMIZER),
    # 20M Improved Mamba-3
    "quick-20m": ExperimentConfig(
        name="quick-20m",
        model=IMPROVED_MAMBA3_20M_MODEL,
        data=DataConfig(dataset_name="tiny_stories", tokenizer_name="byte", max_seq_len=256, batch_size=16, max_eval_batches=50),
        training=TrainingConfig(max_steps=500, eval_interval=100, log_interval=5, save_interval=250, mixed_precision=True),
        optimizer=MUON_20M_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=50),
    ),
    "full-20m": ExperimentConfig(
        name="full-20m",
        model=IMPROVED_MAMBA3_20M_MODEL,
        data=DATA_20M,
        training=TRAINING_20M,
        optimizer=MUON_20M_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=200),
        stages=deepcopy(STAGES_20M),
    ),
    # Production 30M
    "production-quick": ExperimentConfig(
        name="production-quick",
        model=PRODUCTION_MODEL,
        data=DataConfig(dataset_name="tiny_stories", tokenizer_name="byte", max_seq_len=256, batch_size=12, max_eval_batches=50),
        training=TrainingConfig(max_steps=500, eval_interval=100, log_interval=5, save_interval=250),
        optimizer=MUON_PRODUCTION_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=50),
    ),
    "production-full": ExperimentConfig(
        name="production-full",
        model=PRODUCTION_MODEL,
        data=PRODUCTION_DATA,
        training=PRODUCTION_TRAINING,
        optimizer=MUON_PRODUCTION_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=500),
        stages=deepcopy(PRODUCTION_STAGES),
    ),
    # Transformer 10M
    "quick-transformer-10m": ExperimentConfig(
        name="quick-transformer-10m",
        model=TRANSFORMER_10M_MODEL,
        data=DataConfig(dataset_name="tiny_stories", tokenizer_name="byte", max_seq_len=256, batch_size=16, max_eval_batches=50),
        training=TrainingConfig(max_steps=500, eval_interval=100, log_interval=5, save_interval=250, multi_token=True, multi_token_n_ahead=4),
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=50),
    ),
    "full-transformer-10m": ExperimentConfig(
        name="full-transformer-10m",
        model=TRANSFORMER_10M_MODEL,
        data=DATA_20M,
        training=TrainingConfig(
            max_steps=20_000, eval_interval=500, log_interval=10, save_interval=2000,
            multi_token=True, multi_token_n_ahead=4,
        ),
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=200),
        stages=deepcopy(STAGES_20M),
    ),
    # MoE Transformer (sparse MoE FFN, 4 experts top-1)
    "quick-moe-2m": ExperimentConfig(
        name="quick-moe-2m",
        model=MOE_2M_MODEL,
        data=DataConfig(dataset_name="tiny_stories", tokenizer_name="byte", max_seq_len=128, batch_size=32, max_eval_batches=50, num_workers=2),
        training=TrainingConfig(max_steps=500, eval_interval=100, log_interval=5, save_interval=250, mixed_precision=True, multi_token=True, multi_token_n_ahead=4),
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=50),
    ),
    "full-moe-2m": ExperimentConfig(
        name="full-moe-2m",
        model=MOE_2M_MODEL,
        data=DataConfig(dataset_name="tiny_stories", tokenizer_name="byte", max_seq_len=256, batch_size=32, max_eval_batches=50, num_workers=2),
        training=TrainingConfig(
            max_steps=5050, eval_interval=100, log_interval=10, save_interval=500,
            mixed_precision=True, multi_token=True, multi_token_n_ahead=4,
        ),
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=100),
        stages=deepcopy(FULL_STAGES),
    ),
    # Modern Transformer 2M (LLaMA-style: RoPE, RMSNorm, SwiGLU, GQA)
    "quick-modern-2m": ExperimentConfig(
        name="quick-modern-2m",
        model=MODERN_2M_MODEL,
        data=DataConfig(dataset_name="tiny_stories", tokenizer_name="byte", max_seq_len=128, batch_size=32, max_eval_batches=50, num_workers=2),
        training=TrainingConfig(max_steps=500, eval_interval=100, log_interval=5, save_interval=250, mixed_precision=True, multi_token=True, multi_token_n_ahead=4),
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=50),
    ),
    "full-modern-2m": ExperimentConfig(
        name="full-modern-2m",
        model=MODERN_2M_MODEL,
        data=DataConfig(dataset_name="tiny_stories", tokenizer_name="byte", max_seq_len=256, batch_size=32, max_eval_batches=50, num_workers=2),
        training=TrainingConfig(
            max_steps=5050, eval_interval=100, log_interval=10, save_interval=500,
            mixed_precision=True, multi_token=True, multi_token_n_ahead=4,
        ),
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=100),
        stages=deepcopy(FULL_STAGES),
    ),
    # Modern Transformer 10M (LLaMA-style: RoPE, RMSNorm, SwiGLU, GQA)
    "quick-modern-10m": ExperimentConfig(
        name="quick-modern-10m",
        model=MODERN_10M_MODEL,
        data=DataConfig(dataset_name="tiny_stories", tokenizer_name="byte", max_seq_len=256, batch_size=16, max_eval_batches=50, num_workers=2),
        training=TrainingConfig(max_steps=500, eval_interval=100, log_interval=5, save_interval=250, mixed_precision=True, multi_token=True, multi_token_n_ahead=4),
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=50),
    ),
    "full-modern-10m": ExperimentConfig(
        name="full-modern-10m",
        model=MODERN_10M_MODEL,
        data=DataConfig(dataset_name="tiny_stories", tokenizer_name="byte", max_seq_len=512, batch_size=16, max_eval_batches=50, num_workers=2),
        training=TrainingConfig(
            max_steps=20_000, eval_interval=500, log_interval=10, save_interval=2000,
            mixed_precision=True, multi_token=True, multi_token_n_ahead=4,
        ),
        optimizer=ADAMW_OPTIMIZER,
        scheduler=SchedulerConfig(warmup_steps=200),
        stages=deepcopy(STAGES_20M),
    ),
    # Utility
    "sanity-check": ExperimentConfig(
        name="sanity-check",
        model=TRANSFORMER_MODEL,
        data=DataConfig(dataset_name="tiny_stories", tokenizer_name="byte", max_seq_len=32, batch_size=32, max_eval_batches=50),
        training=TrainingConfig(max_steps=100, eval_interval=10, log_interval=5, save_interval=100),
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

# Human-readable labels for the frontend
PRESET_LABELS: dict[str, str] = {
    "quick-transformer": "Transformer Quick (0.86M params, 200 steps)",
    "full-transformer": "Transformer Full (0.86M params, 4 stages, 5K steps)",
    "quick-mamba": "Mamba Quick (0.85M params, 200 steps)",
    "full-mamba": "Mamba Full (0.85M params, 4 stages, 5K steps)",
    "quick-mamba2": "Mamba-2 Quick (0.85M params, 200 steps)",
    "full-mamba2": "Mamba-2 Full (0.85M params, 4 stages, 5K steps)",
    "quick-mamba3": "Mamba-3 Quick (2.14M params, 200 steps)",
    "full-mamba3": "Mamba-3 Full (2.14M params, 4 stages, 5K steps)",
    "quick-improved-mamba3": "Improved Mamba-3 Quick (2.14M params, Muon, 200 steps)",
    "full-improved-mamba3": "Improved Mamba-3 Full (2.14M params, Muon, 4 stages, 5K steps)",
    "quick-20m": "20M Quick (19.4M params, Muon, 500 steps)",
    "full-20m": "20M Full (19.4M params, Muon, 4 stages, 20K steps)",
    "production-quick": "Production Quick (~30M params, Muon, 500 steps)",
    "production-full": "Production Full (~30M params, Muon, 4 stages, 50K steps)",
    "quick-transformer-10m": "Transformer 10M Quick (~10M params, multi-token, 500 steps)",
    "full-transformer-10m": "Transformer 10M Full (~10M params, multi-token, 4 stages, 20K steps)",
    "quick-moe-2m": "MoE 2M Quick (~5M total/~1.5M active, 4 experts top-1, 500 steps)",
    "full-moe-2m": "MoE 2M Full (~5M total/~1.5M active, 4 experts top-1, 4 stages, 5K steps)",
    "quick-modern-2m": "Modern 2M Quick (2M params, RoPE+GQA+SwiGLU, multi-token, 500 steps)",
    "full-modern-2m": "Modern 2M Full (2M params, RoPE+GQA+SwiGLU, multi-token, 4 stages, 5K steps)",
    "quick-modern-10m": "Modern 10M Quick (8.8M params, RoPE+GQA+SwiGLU, multi-token, 500 steps)",
    "full-modern-10m": "Modern 10M Full (8.8M params, RoPE+GQA+SwiGLU, multi-token, 4 stages, 20K steps)",
    "sanity-check": "Sanity Check (0.86M params, overfit 1 batch)",
}


def describe_preset(config: ExperimentConfig) -> str:
    """Auto-compute a human-readable description from a preset's config."""
    stages = config.stages
    if not stages:
        total = config.training.max_steps
        dataset = config.data.dataset_name
        return f"{total:,} steps on {dataset}"

    total = sum(s.max_steps for s in stages)
    parts = [f"{s.name} ({s.max_steps:,})" for s in stages]
    return f"{total:,} steps: {' → '.join(parts)}"


def get_presets() -> list[dict[str, str]]:
    """Return list of {name, label, description} for all presets."""
    return [
        {"name": k, "label": PRESET_LABELS.get(k, k), "description": describe_preset(v)}
        for k, v in PRESETS.items()
    ]


def get_preset(name: str) -> ExperimentConfig | None:
    """Return a preset config by name, or None if not found."""
    return PRESETS.get(name)
