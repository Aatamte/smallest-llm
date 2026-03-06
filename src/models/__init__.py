"""Model registry for easy swapping between architectures."""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

# Registry: name -> (class, default_extra_args)
_REGISTRY: dict[str, tuple[type[nn.Module], dict[str, Any]]] = {}


def register_model(
    name: str,
    cls: type[nn.Module],
    defaults: dict[str, Any] | None = None,
):
    """Register a model class under a given name."""
    _REGISTRY[name] = (cls, defaults or {})


def list_models() -> list[str]:
    """Return all registered model names."""
    return sorted(_REGISTRY.keys())


def build_model(name: str, *, vocab_size: int, max_seq_len: int, extra_args: dict[str, Any] | None = None) -> nn.Module:
    """Build a model by name, merging defaults with extra_args."""
    if name not in _REGISTRY:
        available = ", ".join(list_models())
        raise ValueError(f"Unknown model: {name!r}. Available: {available}")

    cls, defaults = _REGISTRY[name]
    args = {**defaults, **( extra_args or {})}

    # All models take vocab_size; most take max_seq_len
    return cls(vocab_size=vocab_size, max_seq_len=max_seq_len, **args)


# ── Auto-register all built-in models ──────────────────────────────

register_model("transformer",
    cls=None,  # lazy placeholder, replaced below
    defaults={"d_model": 64, "n_heads": 2, "n_layers": 2, "dropout": 0.1},
)

def _register_builtins():
    """Lazy imports to avoid circular deps and keep startup fast."""
    from src.models.tiny_transformer import TinyTransformer
    from src.models.modern_transformer import ModernTransformer
    from src.models.mamba import TinyMamba
    from src.models.mamba2 import TinyMamba2
    from src.models.mamba3 import TinyMamba3
    from src.models.improved_mamba3 import TinyImprovedMamba3
    from src.models.plastic_mamba3 import TinyPlasticMamba3
    from src.models.multiscale_mamba3 import TinyMultiScaleMamba3
    from src.models.hybrid_mamba3 import TinyHybridMamba3
    from src.models.recursive_mamba import RecursiveMamba
    from src.models.moe_transformer import MoETransformer

    _REGISTRY["transformer"] = (TinyTransformer, {
        "d_model": 64, "n_heads": 2, "n_layers": 2, "dropout": 0.1,
    })
    _REGISTRY["modern_transformer"] = (ModernTransformer, {
        "d_model": 256, "n_heads": 8, "n_kv_heads": 4, "n_layers": 12,
        "mlp_ratio": 2.67, "dropout": 0.0,
    })
    _REGISTRY["mamba"] = (TinyMamba, {
        "d_model": 128, "n_layers": 7, "d_state": 16, "d_conv": 4,
        "expand_factor": 2, "dt_rank": 0, "dropout": 0.1,
    })
    _REGISTRY["mamba2"] = (TinyMamba2, {
        "d_model": 128, "n_layers": 7, "n_heads": 0, "d_state": 16,
        "d_conv": 4, "expand_factor": 2, "chunk_size": 64, "dropout": 0.1,
    })
    _REGISTRY["mamba3"] = (TinyMamba3, {
        "d_model": 128, "n_layers": 7, "n_heads": 0, "d_state": 16,
        "expand_factor": 2, "chunk_size": 64, "mlp_factor": 4, "dropout": 0.1,
    })
    _REGISTRY["improved_mamba3"] = (TinyImprovedMamba3, {
        "d_model": 128, "n_layers": 7, "n_heads": 0, "d_state": 16,
        "expand_factor": 2, "chunk_size": 64, "mlp_factor": 4, "dropout": 0.1,
        "gradient_checkpointing": False,
    })
    _REGISTRY["plastic_mamba3"] = (TinyPlasticMamba3, {
        "d_model": 128, "n_layers": 7, "n_heads": 0, "d_state": 16,
        "expand_factor": 2, "chunk_size": 64, "mlp_factor": 4, "dropout": 0.1,
        "gradient_checkpointing": False, "d_plastic": 16,
    })
    _REGISTRY["multiscale_mamba3"] = (TinyMultiScaleMamba3, {
        "d_model": 128, "n_layers": 7, "n_heads": 0, "d_state": 16,
        "expand_factor": 2, "chunk_size": 64, "mlp_factor": 4, "dropout": 0.1,
        "gradient_checkpointing": False,
    })
    _REGISTRY["hybrid_mamba3"] = (TinyHybridMamba3, {
        "d_model": 128, "n_layers": 7, "n_heads": 0, "d_state": 16,
        "expand_factor": 2, "chunk_size": 64, "mlp_factor": 4, "dropout": 0.1,
        "gradient_checkpointing": False, "attn_layer_idx": -1, "attn_window": 64,
    })
    _REGISTRY["moe_transformer"] = (MoETransformer, {
        "d_model": 128, "n_heads": 4, "n_kv_heads": 2, "n_layers": 8,
        "mlp_ratio": 2.67, "n_experts": 4, "top_k": 1,
        "aux_loss_weight": 0.01, "dropout": 0.0,
    })
    _REGISTRY["recursive_mamba"] = (RecursiveMamba, {
        "d_model": 128, "n_layers": 2, "n_heads": 0, "d_state": 16,
        "expand_factor": 2, "chunk_size": 64, "mlp_factor": 4, "dropout": 0.1,
        "gradient_checkpointing": False,
        "n_recursions": 6, "n_supervision": 3, "deep_sup_weight": 1.0,
    })


# Clear the placeholder and do the real registration
_REGISTRY.clear()
_register_builtins()
