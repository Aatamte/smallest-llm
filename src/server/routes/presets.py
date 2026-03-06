"""Config and eval preset endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/api/config")
def get_config():
    """Return the default experiment config."""
    from src.config.base import ExperimentConfig
    return ExperimentConfig().to_dict()


@router.get("/api/presets")
def list_presets():
    """Return list of available config presets."""
    from src.config.presets import get_presets
    return get_presets()


@router.get("/api/presets/{name}")
def get_preset_config(name: str):
    """Return full config dict for a named preset."""
    from src.config.presets import get_preset
    preset = get_preset(name)
    if preset is None:
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")
    return preset.to_dict()


@router.get("/api/flops-budgets")
def list_flops_budgets():
    """Return list of available FLOPs budget options."""
    from src.config.presets import get_flops_budgets
    return get_flops_budgets()
