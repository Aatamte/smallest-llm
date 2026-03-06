"""Tests for tokenizer and data pipeline."""

import torch

from src.config.base import DataConfig
from src.data.text import TextDataset, build_dataloaders
from src.data.tokenizer import CharTokenizer, build_tokenizer


def test_char_tokenizer_encode_decode():
    text = "hello world"
    tok = CharTokenizer(text)
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_char_tokenizer_vocab():
    text = "abcabc"
    tok = CharTokenizer(text)
    assert tok.vocab_size == 3
    assert set(tok.char_to_id.keys()) == {"a", "b", "c"}


def test_char_tokenizer_save_load(tmp_path):
    text = "hello world"
    tok = CharTokenizer(text)
    path = str(tmp_path / "tok.json")
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.vocab_size == tok.vocab_size
    assert loaded.encode("hello") == tok.encode("hello")


def test_build_tokenizer():
    tok = build_tokenizer("char", "abc")
    assert tok.vocab_size == 3


def test_text_dataset_shapes():
    tokens = torch.arange(100)
    ds = TextDataset(tokens, seq_len=10)
    assert len(ds) == 9  # (100 - 1) // 10
    sample = ds[0]
    assert sample["input_ids"].shape == (10,)
    assert sample["labels"].shape == (10,)
    # labels are shifted by 1
    assert torch.equal(sample["input_ids"], torch.arange(0, 10))
    assert torch.equal(sample["labels"], torch.arange(1, 11))


def test_build_dataloaders():
    text = "abcdefghij" * 100
    config = DataConfig(batch_size=4, max_seq_len=8, num_workers=0)
    tok = build_tokenizer("char", text)
    train_loader, val_loader, test_loader = build_dataloaders(config, text, tok)
    batch = next(iter(train_loader))
    assert batch["input_ids"].shape == (4, 8)
    assert batch["labels"].shape == (4, 8)
