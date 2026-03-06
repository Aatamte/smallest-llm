"""Dataset downloading and caching."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import requests

# Use local cache only — never hit HuggingFace servers during training
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

HF_CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "hf_cache"

_TEXT_DATASETS = {
    "tiny_shakespeare": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
}


@dataclass
class TextFileDataset:
    """In-memory text dataset (for small datasets)."""
    text: str


@dataclass
class HFStreamingDataset:
    """Reference to a HuggingFace dataset for streaming."""
    path: str
    text_field: str
    train_split: str
    val_split: str


_HF_DATASETS: dict[str, HFStreamingDataset] = {
    "tiny_stories": HFStreamingDataset(
        path="roneneldan/TinyStories",
        text_field="text",
        train_split="train",
        val_split="validation",
    ),
    "minipile": HFStreamingDataset(
        path="JeanKaddour/minipile",
        text_field="text",
        train_split="train",
        val_split="validation",
    ),
    "openwebtext": HFStreamingDataset(
        path="Skylion007/openwebtext",
        text_field="text",
        train_split="train",
        val_split="train",  # no dedicated val split
    ),
}

@dataclass
class HFChatDataset:
    """Reference to a HuggingFace chat/instruction dataset."""
    path: str
    train_split: str
    val_split: str


_HF_CHAT_DATASETS: dict[str, HFChatDataset] = {
    "ultrachat": HFChatDataset(
        path="HuggingFaceH4/ultrachat_200k",
        train_split="train_sft",
        val_split="test_sft",
    ),
}

DatasetResult = Union[TextFileDataset, HFStreamingDataset, HFChatDataset]


def load_dataset(name: str, cache_dir: str = "data") -> DatasetResult:
    """Load a dataset by name. Returns either in-memory text, streaming, or chat reference."""
    if name in _HF_CHAT_DATASETS:
        return _HF_CHAT_DATASETS[name]

    if name in _HF_DATASETS:
        return _HF_DATASETS[name]

    if name in _TEXT_DATASETS:
        cache_path = Path(cache_dir) / f"{name}.txt"
        if cache_path.exists():
            return TextFileDataset(cache_path.read_text(encoding="utf-8"))

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(_TEXT_DATASETS[name], timeout=30)
        resp.raise_for_status()
        cache_path.write_text(resp.text, encoding="utf-8")
        return TextFileDataset(resp.text)

    available = list(_TEXT_DATASETS) + list(_HF_DATASETS) + list(_HF_CHAT_DATASETS)
    raise ValueError(f"Unknown dataset: {name!r}. Available: {available}")
