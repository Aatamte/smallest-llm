"""Export a trained model to HuggingFace-style directory."""

from __future__ import annotations

import json
import os

import torch
from safetensors.torch import save_model

from src.config.base import ExperimentConfig
from src.data.tokenizer import CharTokenizer, HFTokenizer, Tokenizer


def export_model(
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    config: ExperimentConfig,
    save_dir: str,
) -> str:
    """Export model weights + config + tokenizer to a directory.

    Creates:
        save_dir/model.safetensors  — model weights
        save_dir/config.json        — architecture metadata
        save_dir/tokenizer.json     — tokenizer data
        save_dir/experiment_config.json — full training config

    Returns the save_dir path.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Model weights (save_model handles weight-tied models correctly)
    save_model(model, os.path.join(save_dir, "model.safetensors"))

    # Architecture config
    model_config = {
        "model_type": config.model.name,
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": config.data.max_seq_len,
        **config.model.extra_args,
    }
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)

    # Tokenizer
    if isinstance(tokenizer, CharTokenizer):
        tokenizer.save(os.path.join(save_dir, "tokenizer.json"))
    elif isinstance(tokenizer, HFTokenizer):
        with open(os.path.join(save_dir, "tokenizer.json"), "w") as f:
            json.dump({"type": "hf", "name": tokenizer._tok.name_or_path}, f, indent=2)

    # Full experiment config for provenance
    with open(os.path.join(save_dir, "experiment_config.json"), "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    return save_dir
