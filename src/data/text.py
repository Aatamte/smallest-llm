"""Text dataset and dataloader construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from src.config.base import DataConfig
    from src.data.tokenizer import Tokenizer


class TextDataset(Dataset):
    """Tokenized text dataset that produces fixed-length sequences."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(0, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = idx * self.seq_len
        x = self.tokens[start : start + self.seq_len]
        y = self.tokens[start + 1 : start + self.seq_len + 1]
        return {"input_ids": x, "labels": y}


def build_dataloaders(
    config: DataConfig,
    text: str,
    tokenizer: Tokenizer,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Tokenize text, split, and build dataloaders."""
    token_ids = tokenizer.encode(text)
    tokens = torch.tensor(token_ids, dtype=torch.long)

    n = len(tokens)
    train_end = int(n * config.train_split)
    val_end = train_end + int(n * config.val_split)

    train_tokens = tokens[:train_end]
    val_tokens = tokens[train_end:val_end]
    test_tokens = tokens[val_end:]

    train_ds = TextDataset(train_tokens, config.max_seq_len)
    val_ds = TextDataset(val_tokens, config.max_seq_len)
    test_ds = TextDataset(test_tokens, config.max_seq_len)

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=False,
        drop_last=True,
    )

    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
