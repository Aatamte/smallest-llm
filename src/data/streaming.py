"""Streaming dataset for HuggingFace datasets."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

# Force offline mode when cache exists — avoids network timeouts
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

if TYPE_CHECKING:
    from src.config.base import DataConfig
    from src.data.datasets import HFStreamingDataset
    from src.data.tokenizer import Tokenizer


class StreamingTextDataset(IterableDataset):
    """Streams text from an HF dataset, tokenizes on-the-fly, yields fixed-length sequences."""

    def __init__(
        self,
        hf_path: str,
        split: str,
        text_field: str,
        tokenizer: Tokenizer,
        seq_len: int,
        shuffle_buffer: int = 10_000,
        config: str | None = None,
    ):
        self.hf_path = hf_path
        self.split = split
        self.text_field = text_field
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer
        self.config = config
        self._cached_ds = None

    def _get_dataset(self):
        """Load dataset once and cache it."""
        if self._cached_ds is None:
            import logging
            from datasets import load_dataset
            from src.data.datasets import HF_CACHE_DIR

            # Suppress noisy HF cache messages
            hf_logger = logging.getLogger("datasets.builder")
            prev_level = hf_logger.level
            hf_logger.setLevel(logging.ERROR)

            cache_dir = str(HF_CACHE_DIR)
            try:
                self._cached_ds = load_dataset(self.hf_path, self.config, split=self.split, cache_dir=cache_dir)
            except Exception:
                self._cached_ds = load_dataset(self.hf_path, self.config, split=self.split, streaming=True)
            finally:
                hf_logger.setLevel(prev_level)
        return self._cached_ds

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        ds = self._get_dataset()
        if self.shuffle_buffer > 0:
            try:
                ds = ds.shuffle(buffer_size=self.shuffle_buffer)
            except TypeError:
                ds = ds.shuffle()

        token_buffer: list[int] = []

        for example in ds:
            text = example[self.text_field]
            tokens = self.tokenizer.encode(text)
            token_buffer.extend(tokens)

            while len(token_buffer) >= self.seq_len + 1:
                chunk = token_buffer[: self.seq_len + 1]
                token_buffer = token_buffer[self.seq_len + 1 :]

                t = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": t[:-1], "labels": t[1:]}


def build_streaming_dataloaders(
    config: DataConfig,
    dataset_info: HFStreamingDataset,
    tokenizer: Tokenizer,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from a streaming HF dataset."""
    train_ds = StreamingTextDataset(
        hf_path=dataset_info.path,
        split=dataset_info.train_split,
        text_field=dataset_info.text_field,
        tokenizer=tokenizer,
        seq_len=config.max_seq_len,
        config=dataset_info.config,
    )

    val_ds = StreamingTextDataset(
        hf_path=dataset_info.path,
        split=dataset_info.val_split,
        text_field=dataset_info.text_field,
        tokenizer=tokenizer,
        seq_len=config.max_seq_len,
        shuffle_buffer=0,  # no shuffling for deterministic eval
        config=dataset_info.config,
    )

    # Reuse validation as test (TinyStories only has train/validation)
    test_ds = StreamingTextDataset(
        hf_path=dataset_info.path,
        split=dataset_info.val_split,
        text_field=dataset_info.text_field,
        tokenizer=tokenizer,
        seq_len=config.max_seq_len,
        shuffle_buffer=0,
        config=dataset_info.config,
    )

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=False,
    )

    return (
        DataLoader(train_ds, **loader_kwargs),
        DataLoader(val_ds, **loader_kwargs),
        DataLoader(test_ds, **loader_kwargs),
    )
