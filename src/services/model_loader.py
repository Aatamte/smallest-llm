"""Shared model loading helpers used by eval and chat services."""

from __future__ import annotations

import os


def is_custom_export(model_dir: str) -> bool:
    """Check if a model dir uses our custom export format vs standard HF format."""
    return os.path.exists(os.path.join(model_dir, "experiment_config.json"))


def load_eval_model(model_dir: str):
    """Load an eval model from a local directory (custom export or HF format)."""
    if is_custom_export(model_dir):
        from src.evaluation.exported_model import ExportedModel
        return ExportedModel(model_dir)
    else:
        from src.evaluation.hf_model import HFModel
        return HFModel(model_dir)
