"""Chat/SFT data pipeline: template formatting, loss masking, and streaming."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

if TYPE_CHECKING:
    from src.config.base import DataConfig
    from src.data.datasets import HFChatDataset
    from src.data.tokenizer import HFTokenizer


IGNORE_INDEX = -100


class ChatTemplate:
    """Formats conversations into token sequences with loss masking.

    Template: <|user|>{message}<|assistant|>{response}<|end|>
    Loss is only computed on assistant response tokens (shifted by 1).
    """

    def __init__(self, tokenizer: HFTokenizer):
        self.tokenizer = tokenizer
        self.user_token_id = tokenizer.token_to_id("<|user|>")
        self.assistant_token_id = tokenizer.token_to_id("<|assistant|>")
        self.end_token_id = tokenizer.token_to_id("<|end|>")

    def format_conversation(
        self, messages: list[dict[str, str]], max_seq_len: int
    ) -> dict[str, torch.Tensor] | None:
        """Format a conversation into input_ids and labels with loss masking.

        Returns None if the conversation is empty after tokenization.
        Labels are set to IGNORE_INDEX (-100) for all non-assistant tokens.
        """
        all_token_ids: list[int] = []
        all_labels: list[int] = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "user":
                # <|user|> + content tokens — all masked
                prefix_ids = [self.user_token_id]
                content_ids = self.tokenizer.encode(content)
                segment_ids = prefix_ids + content_ids
                segment_labels = [IGNORE_INDEX] * len(segment_ids)
            elif role == "assistant":
                # <|assistant|> + content tokens + <|end|>
                # Mask the <|assistant|> token, train on content + <|end|>
                prefix_ids = [self.assistant_token_id]
                content_ids = self.tokenizer.encode(content)
                end_ids = [self.end_token_id]
                segment_ids = prefix_ids + content_ids + end_ids
                segment_labels = (
                    [IGNORE_INDEX]  # mask <|assistant|> marker
                    + content_ids   # train on content
                    + end_ids       # train on <|end|>
                )
            else:
                # Skip system or other roles
                continue

            all_token_ids.extend(segment_ids)
            all_labels.extend(segment_labels)

        if not all_token_ids:
            return None

        # Truncate to max_seq_len + 1 (need +1 for shifted labels)
        max_len = max_seq_len + 1
        all_token_ids = all_token_ids[:max_len]
        all_labels = all_labels[:max_len]

        # Pad if shorter
        pad_len = max_len - len(all_token_ids)
        if pad_len > 0:
            all_token_ids.extend([0] * pad_len)
            all_labels.extend([IGNORE_INDEX] * pad_len)

        # Create input_ids (first max_seq_len tokens) and labels (shifted by 1)
        input_ids = torch.tensor(all_token_ids[:max_seq_len], dtype=torch.long)
        labels = torch.tensor(all_labels[1:max_seq_len + 1], dtype=torch.long)

        return {"input_ids": input_ids, "labels": labels}


class StreamingChatDataset(IterableDataset):
    """Streams conversations from an HF chat dataset, applies chat template."""

    def __init__(
        self,
        hf_path: str,
        split: str,
        tokenizer: HFTokenizer,
        chat_template: ChatTemplate,
        seq_len: int,
        shuffle_buffer: int = 10_000,
    ):
        self.hf_path = hf_path
        self.split = split
        self.tokenizer = tokenizer
        self.chat_template = chat_template
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        from datasets import load_dataset
        from src.data.datasets import HF_CACHE_DIR

        cache_dir = str(HF_CACHE_DIR)
        try:
            ds = load_dataset(self.hf_path, split=self.split, cache_dir=cache_dir)
        except Exception:
            # Fallback to streaming if cache download fails
            ds = load_dataset(self.hf_path, split=self.split, streaming=True)
        if self.shuffle_buffer > 0:
            try:
                ds = ds.shuffle(buffer_size=self.shuffle_buffer)
            except TypeError:
                # Map-style dataset doesn't accept buffer_size
                ds = ds.shuffle()

        for example in ds:
            messages = example.get("messages", [])
            if not messages:
                continue

            result = self.chat_template.format_conversation(messages, self.seq_len)
            if result is not None:
                yield result


def build_chat_dataloaders(
    config: DataConfig,
    dataset_info: HFChatDataset,
    tokenizer: HFTokenizer,
    chat_template: ChatTemplate,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test DataLoaders from a streaming HF chat dataset."""
    train_ds = StreamingChatDataset(
        hf_path=dataset_info.path,
        split=dataset_info.train_split,
        tokenizer=tokenizer,
        chat_template=chat_template,
        seq_len=config.max_seq_len,
    )

    val_ds = StreamingChatDataset(
        hf_path=dataset_info.path,
        split=dataset_info.val_split,
        tokenizer=tokenizer,
        chat_template=chat_template,
        seq_len=config.max_seq_len,
        shuffle_buffer=0,
    )

    # Reuse val as test
    test_ds = StreamingChatDataset(
        hf_path=dataset_info.path,
        split=dataset_info.val_split,
        tokenizer=tokenizer,
        chat_template=chat_template,
        seq_len=config.max_seq_len,
        shuffle_buffer=0,
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
