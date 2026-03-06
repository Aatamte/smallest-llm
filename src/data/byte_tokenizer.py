"""Byte-level tokenizer — encodes text as raw UTF-8 bytes.

Universal: handles any input text without training data dependency.
Vocab size is always 256.
"""

from __future__ import annotations


class ByteTokenizer:
    """Encodes text as UTF-8 byte sequences."""

    vocab_size: int = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return bytes(ids).decode("utf-8", errors="replace")
