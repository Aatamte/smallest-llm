"""Tests for chat/SFT data pipeline."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.base import ExperimentConfig, StageConfig
from src.config.presets import get_preset
from src.data.chat import IGNORE_INDEX, ChatTemplate, StreamingChatDataset
from src.data.datasets import HFChatDataset
from src.data.tokenizer import CHAT_SPECIAL_TOKENS, HFTokenizer
from src.models.base import BaseModel, ModelOutput
from src.training.pipeline import PipelineRunner, _resize_embeddings


# --- Helpers ---

def _make_tokenizer():
    """Build a GPT-2 tokenizer with chat tokens added."""
    tok = HFTokenizer("gpt2")
    tok.add_chat_tokens()
    return tok


def _make_messages():
    """Simple single-turn conversation."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


def _make_multi_turn_messages():
    """Multi-turn conversation."""
    return [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "6"},
    ]


# --- ChatTemplate tests ---

class TestChatTemplate:
    def test_format_produces_correct_shapes(self):
        tok = _make_tokenizer()
        template = ChatTemplate(tok)
        result = template.format_conversation(_make_messages(), max_seq_len=64)

        assert result is not None
        assert result["input_ids"].shape == (64,)
        assert result["labels"].shape == (64,)

    def test_user_tokens_are_masked(self):
        tok = _make_tokenizer()
        template = ChatTemplate(tok)
        result = template.format_conversation(_make_messages(), max_seq_len=64)

        # The first tokens should be <|user|> + "Hello" — all masked
        labels = result["labels"]
        # At least the first few labels (corresponding to user content) should be -100
        # Find where the assistant content starts
        input_ids = result["input_ids"]
        assistant_id = tok.token_to_id("<|assistant|>")

        # Everything before <|assistant|> in labels should be masked
        assistant_positions = (input_ids == assistant_id).nonzero(as_tuple=True)[0]
        assert len(assistant_positions) > 0
        first_assistant_pos = assistant_positions[0].item()

        # Labels up to and including the position after <|assistant|> marker should be masked
        # (because labels are shifted by 1, the label at first_assistant_pos corresponds to
        # the token AFTER <|assistant|>)
        for i in range(first_assistant_pos):
            assert labels[i].item() == IGNORE_INDEX, f"Label at position {i} should be masked"

    def test_assistant_tokens_are_trained(self):
        tok = _make_tokenizer()
        template = ChatTemplate(tok)
        result = template.format_conversation(_make_messages(), max_seq_len=64)

        labels = result["labels"]
        # There should be some non-IGNORE_INDEX labels (the assistant content)
        trained_mask = labels != IGNORE_INDEX
        assert trained_mask.any(), "Should have some trainable (non-masked) labels"

    def test_empty_messages_returns_none(self):
        tok = _make_tokenizer()
        template = ChatTemplate(tok)
        result = template.format_conversation([], max_seq_len=64)
        assert result is None

    def test_multi_turn_has_multiple_trained_regions(self):
        tok = _make_tokenizer()
        template = ChatTemplate(tok)
        result = template.format_conversation(_make_multi_turn_messages(), max_seq_len=128)

        assert result is not None
        labels = result["labels"]

        # Find transitions from masked to unmasked — should happen at least twice
        # (once for each assistant turn)
        transitions = 0
        prev_masked = True
        for label in labels:
            is_masked = label.item() == IGNORE_INDEX
            if prev_masked and not is_masked:
                transitions += 1
            prev_masked = is_masked

        assert transitions >= 2, f"Expected at least 2 trained regions, got {transitions}"

    def test_truncation_to_max_seq_len(self):
        tok = _make_tokenizer()
        template = ChatTemplate(tok)
        # Very short seq_len should still produce correct shapes
        result = template.format_conversation(_make_messages(), max_seq_len=8)
        assert result is not None
        assert result["input_ids"].shape == (8,)
        assert result["labels"].shape == (8,)

    def test_padding_short_conversations(self):
        tok = _make_tokenizer()
        template = ChatTemplate(tok)
        result = template.format_conversation(_make_messages(), max_seq_len=256)

        assert result is not None
        labels = result["labels"]
        # Padding positions should be masked
        # The conversation is short, so most of the 256 positions should be padded/masked
        masked_count = (labels == IGNORE_INDEX).sum().item()
        assert masked_count > 200, f"Expected most positions to be masked for a short conversation"

    def test_system_role_is_skipped(self):
        tok = _make_tokenizer()
        template = ChatTemplate(tok)
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = template.format_conversation(messages, max_seq_len=64)
        assert result is not None
        # Should still have trainable labels (system is skipped, not breaking)
        assert (result["labels"] != IGNORE_INDEX).any()


# --- Embedding resize tests ---

class TestResizeEmbeddings:
    def test_resize_increases_vocab(self):
        model = _make_tiny_model(vocab_size=100, d_model=16)
        _resize_embeddings(model, 110)
        assert model.token_emb.weight.shape[0] == 110
        assert model.head.weight.shape[0] == 110

    def test_resize_preserves_existing_weights(self):
        model = _make_tiny_model(vocab_size=100, d_model=16)
        old_weights = model.token_emb.weight[:100].clone()
        _resize_embeddings(model, 110)
        torch.testing.assert_close(model.token_emb.weight[:100], old_weights)

    def test_resize_noop_when_same_size(self):
        model = _make_tiny_model(vocab_size=100, d_model=16)
        old_weight_ptr = model.token_emb.weight.data_ptr()
        _resize_embeddings(model, 100)
        # Should not have changed anything
        assert model.token_emb.weight.data_ptr() == old_weight_ptr

    def test_resize_handles_weight_tied_models(self):
        model = _make_tiny_model(vocab_size=100, d_model=16, tie_weights=True)
        _resize_embeddings(model, 110)
        # After resize, weights should still be tied
        assert model.head.weight.data_ptr() == model.token_emb.weight.data_ptr()

    def test_resize_handles_untied_models(self):
        model = _make_tiny_model(vocab_size=100, d_model=16, tie_weights=False)
        old_head_weights = model.head.weight[:100].clone()
        _resize_embeddings(model, 110)
        assert model.head.weight.shape[0] == 110
        torch.testing.assert_close(model.head.weight[:100], old_head_weights)
        # Untied: different data pointers
        assert model.head.weight.data_ptr() != model.token_emb.weight.data_ptr()


# --- Preset tests ---

def test_stage_type_sft_in_stage_config():
    """StageConfig supports stage_type='sft' with a dataset_name."""
    stage = StageConfig(name="sft", max_steps=50, stage_type="sft", dataset_name="ultrachat")
    assert stage.stage_type == "sft"
    assert stage.dataset_name == "ultrachat"


def test_stage_type_serialization_roundtrip():
    """stage_type survives to_dict / from_dict round-trip."""
    config = ExperimentConfig(
        name="test",
        stages=[
            StageConfig(name="pretrain", max_steps=100, stage_type="pretrain"),
            StageConfig(name="sft", max_steps=50, stage_type="sft", dataset_name="ultrachat"),
        ],
    )
    d = config.to_dict()
    restored = ExperimentConfig.from_dict(d)

    assert restored.stages is not None
    assert restored.stages[0].stage_type == "pretrain"
    assert restored.stages[1].stage_type == "sft"
    assert restored.stages[1].dataset_name == "ultrachat"


# --- Tokenizer tests ---

def test_add_chat_tokens():
    tok = HFTokenizer("gpt2")
    old_vocab = tok.vocab_size
    num_added = tok.add_chat_tokens()
    assert num_added == 3
    assert tok.vocab_size == old_vocab + 3

    # Check tokens are resolvable
    for token in CHAT_SPECIAL_TOKENS:
        token_id = tok.token_to_id(token)
        assert token_id >= old_vocab


def test_add_chat_tokens_idempotent():
    tok = HFTokenizer("gpt2")
    tok.add_chat_tokens()
    size_after_first = tok.vocab_size
    tok.add_chat_tokens()
    # Second call should add 0 tokens
    assert tok.vocab_size == size_after_first


# --- Helper model for tests ---

class TinyTestModel(BaseModel):
    def __init__(self, vocab_size, d_model=16, tie_weights=True):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.head.weight = self.token_emb.weight
        self.max_seq_len = 256

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.token_emb(input_ids)
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=IGNORE_INDEX,
            )
        return ModelOutput(loss=loss, logits=logits)


def _make_tiny_model(vocab_size=100, d_model=16, tie_weights=True):
    return TinyTestModel(vocab_size, d_model, tie_weights)


# --- Loss masking integration test ---

def test_loss_masking_excludes_user_tokens():
    """Verify that labels=-100 actually results in no gradient for those positions."""
    tok = _make_tokenizer()
    template = ChatTemplate(tok)
    model = _make_tiny_model(vocab_size=tok.vocab_size, d_model=16)

    result = template.format_conversation(_make_messages(), max_seq_len=32)
    assert result is not None

    input_ids = result["input_ids"].unsqueeze(0)  # batch dim
    labels = result["labels"].unsqueeze(0)

    output = model(input_ids, labels=labels)
    assert output.loss is not None
    assert output.loss.item() > 0  # Should have some loss from assistant tokens

    # Compare with all-masked labels — PyTorch cross_entropy returns NaN when all
    # labels are ignored (0/0 in mean reduction), which confirms masking works
    all_masked = torch.full_like(labels, IGNORE_INDEX)
    output_masked = model(input_ids, labels=all_masked)
    import math
    assert math.isnan(output_masked.loss.item()), "All-masked labels should produce NaN loss"
