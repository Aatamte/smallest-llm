"""Pre-download HuggingFace datasets for offline training.

Usage:
    uv run python scripts/download_datasets.py --datasets tiny_stories minipile
    uv run python scripts/download_datasets.py --datasets openwebtext
    uv run python scripts/download_datasets.py --all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.datasets import _HF_DATASETS, _HF_CHAT_DATASETS, HF_CACHE_DIR

# Large datasets that need special handling (download raw files first, then process)
_LARGE_DATASETS = {"openwebtext"}


def download_hf_dataset(name: str, path: str, splits: list[str]) -> None:
    """Download a HuggingFace dataset to the local cache."""
    from datasets import load_dataset

    cache_dir = str(HF_CACHE_DIR)
    print(f"\nDownloading {name} ({path})...")

    if name in _LARGE_DATASETS:
        _download_large_dataset(name, path, splits, cache_dir)
        return

    for split in splits:
        print(f"  Split: {split}...", end=" ", flush=True)
        try:
            ds = load_dataset(path, split=split, cache_dir=cache_dir)
            print(f"done ({len(ds):,} examples)")
        except Exception as e:
            print(f"FAILED: {e}")


def _download_large_dataset(name: str, path: str, splits: list[str], cache_dir: str) -> None:
    """Download large datasets using huggingface-cli, then load with num_proc."""
    from datasets import load_dataset

    # Download large datasets directly with load_dataset + multiprocessing
    for split in splits:
        print(f"  Split: {split}...", end=" ", flush=True)
        try:
            ds = load_dataset(path, split=split, cache_dir=cache_dir, num_proc=4)
            print(f"done ({len(ds):,} examples)")
        except Exception as e:
            print(f"FAILED: {e}")


def get_all_dataset_names() -> list[str]:
    return list(_HF_DATASETS.keys()) + list(_HF_CHAT_DATASETS.keys())


def main():
    parser = argparse.ArgumentParser(description="Download HF datasets for offline training")
    parser.add_argument("--datasets", nargs="+", help="Dataset names to download")
    parser.add_argument("--all", action="store_true", help="Download all known datasets")
    parser.add_argument("--cache-dir", type=str, default=None, help="Override cache directory")
    args = parser.parse_args()

    if args.cache_dir:
        import src.data.datasets as ds_mod
        ds_mod.HF_CACHE_DIR = Path(args.cache_dir)

    if args.all:
        names = get_all_dataset_names()
    elif args.datasets:
        names = args.datasets
    else:
        print("Specify --datasets or --all")
        print(f"Available: {get_all_dataset_names()}")
        return

    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cache directory: {HF_CACHE_DIR}")

    for name in names:
        if name in _HF_DATASETS:
            info = _HF_DATASETS[name]
            splits = list({info.train_split, info.val_split})
            download_hf_dataset(name, info.path, splits)
        elif name in _HF_CHAT_DATASETS:
            info = _HF_CHAT_DATASETS[name]
            splits = list({info.train_split, info.val_split})
            download_hf_dataset(name, info.path, splits)
        else:
            print(f"Unknown dataset: {name!r}. Available: {get_all_dataset_names()}")

    # Print cache size
    total = sum(f.stat().st_size for f in HF_CACHE_DIR.rglob("*") if f.is_file())
    print(f"\nTotal cache size: {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
