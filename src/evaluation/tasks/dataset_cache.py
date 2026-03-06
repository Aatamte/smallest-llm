"""Offline-safe dataset loading for eval tasks.

Downloads HF datasets once to data/eval/<name>/ as Arrow files,
then loads from disk on all subsequent runs with zero network calls.
"""
from __future__ import annotations

from pathlib import Path

from datasets import Dataset, load_dataset, load_from_disk


def load_cached_dataset(
    hf_path: str,
    subset: str | None = None,
    split: str = "validation",
    cache_dir: str = "data/eval",
    revision: str | None = None,
) -> Dataset:
    """Load dataset from local cache, downloading on first use.

    Saves to data/eval/<name>_<subset>_<split>/ as Arrow files.
    Subsequent loads are purely offline — no network calls.
    """
    # Build a unique local directory name
    name = hf_path.replace("/", "_")
    if subset:
        name += f"_{subset}"
    name += f"_{split}"
    local_path = Path(cache_dir) / name

    if local_path.exists():
        return load_from_disk(str(local_path))

    # First time: download and save locally
    kwargs = {"split": split}
    if revision:
        kwargs["revision"] = revision

    if subset:
        ds = load_dataset(hf_path, subset, **kwargs)
    else:
        ds = load_dataset(hf_path, **kwargs)

    local_path.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(local_path))
    return ds
