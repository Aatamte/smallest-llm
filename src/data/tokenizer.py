"""Tokenizer protocol and implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


class CharTokenizer:
    """Character-level tokenizer. Zero dependencies, good for debugging."""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.char_to_id = {c: i for i, c in enumerate(chars)}
        self.id_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[c] for c in text if c in self.char_to_id]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char.get(i, "") for i in ids)

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump({"chars": list(self.char_to_id.keys())}, f)

    @classmethod
    def load(cls, path: str) -> CharTokenizer:
        import json
        with open(path) as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        chars = data["chars"]
        tok.char_to_id = {c: i for i, c in enumerate(chars)}
        tok.id_to_char = {i: c for i, c in enumerate(chars)}
        tok.vocab_size = len(chars)
        return tok


CHAT_SPECIAL_TOKENS = ["<|user|>", "<|assistant|>", "<|end|>"]


class HFTokenizer:
    """Wraps a HuggingFace tokenizer to match the Tokenizer protocol."""

    def __init__(self, name: str):
        from transformers import AutoTokenizer
        self._tok = AutoTokenizer.from_pretrained(name)
        # Suppress "Token indices sequence length is longer than..." warning.
        # Our data pipeline handles chunking to max_seq_len independently.
        self._tok.model_max_length = int(1e30)
        self.vocab_size = self._tok.vocab_size

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def token_to_id(self, token: str) -> int:
        """Get the token ID for a special token string."""
        return self._tok.convert_tokens_to_ids(token)

    def add_chat_tokens(self) -> int:
        """Add chat special tokens. Returns number of tokens added."""
        num_added = self._tok.add_special_tokens(
            {"additional_special_tokens": CHAT_SPECIAL_TOKENS}
        )
        self.vocab_size = len(self._tok)
        return num_added


def build_tokenizer(name: str, text: str | None = None) -> Tokenizer:
    """Build a tokenizer by name."""
    if name == "char":
        if text is None:
            raise ValueError("CharTokenizer requires text to build vocabulary")
        return CharTokenizer(text)
    if name == "byte":
        from src.data.byte_tokenizer import ByteTokenizer
        return ByteTokenizer()  # text arg ignored — byte tokenizer is universal
    return HFTokenizer(name)
