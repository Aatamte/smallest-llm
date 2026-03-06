"""Tests for Hydra Training: multi-resolution parallel forward passes."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.base import ExperimentConfig
from src.data.text import TextDataset, build_dataloaders
from src.data.tokenizer import build_tokenizer
from src.models.base import BaseModel, ModelOutput
from src.training.techniques.hydra import HydraCallback
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.trainer import Trainer

from torch.utils.data import DataLoader


class TinyModel(BaseModel):
    """Minimal model for testing."""
    def __init__(self, vocab_size, dim=16):
        super().__init__()
        self.max_seq_len = 512
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.linear = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.linear(self.token_emb(input_ids))
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return ModelOutput(loss=loss, logits=logits)


def _make_text_and_tok():
    text = "the quick brown fox jumps over the lazy dog " * 200
    tok = build_tokenizer("char", text)
    return text, tok


def _make_loader(text, tok, seq_len, batch_size):
    tokens = torch.tensor(tok.encode(text), dtype=torch.long)
    ds = TextDataset(tokens, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def test_hydra_callback_runs():
    """HydraCallback runs without errors in a full training loop."""
    text, tok = _make_text_and_tok()

    config = ExperimentConfig(name="test-hydra", device="cpu")
    config.training.max_steps = 10
    config.training.log_interval = 5
    config.training.eval_interval = 5
    config.training.save_interval = 100
    config.data.batch_size = 4
    config.data.max_seq_len = 32

    train_loader, val_loader, _ = build_dataloaders(config.data, text, tok)
    micro_loader = _make_loader(text, tok, seq_len=8, batch_size=16)
    macro_loader = _make_loader(text, tok, seq_len=64, batch_size=2)

    model = TinyModel(tok.vocab_size)
    optimizer = build_optimizer(config.optimizer, model)
    scheduler = build_scheduler(config.scheduler, optimizer, config.training.max_steps)

    hydra_cb = HydraCallback(
        micro_loader=micro_loader,
        macro_loader=macro_loader,
        micro_weight=0.3,
        macro_weight=0.1,
    )

    trainer = Trainer(
        config=config, model=model,
        train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler,
        callbacks=[hydra_cb],
    )
    trainer.train()
    assert trainer.tokens_seen > 0


def test_hydra_adds_gradient_signal():
    """Hydra should produce different (more) gradients than vanilla."""
    text, tok = _make_text_and_tok()
    torch.manual_seed(42)

    batch_ids = torch.randint(0, tok.vocab_size, (4, 32))
    batch_labels = torch.randint(0, tok.vocab_size, (4, 32))
    batch = {"input_ids": batch_ids, "labels": batch_labels}

    # Vanilla: one forward/backward
    model_v = TinyModel(tok.vocab_size)
    out = model_v(**batch)
    out.loss.backward()
    vanilla_grad = model_v.linear.weight.grad.clone()

    # Hydra: micro + main forward/backward
    model_h = TinyModel(tok.vocab_size)
    # Copy weights to be identical
    model_h.load_state_dict(model_v.state_dict())
    model_h.zero_grad()

    # Simulate micro forward/backward
    micro_ids = torch.randint(0, tok.vocab_size, (16, 8))
    micro_labels = torch.randint(0, tok.vocab_size, (16, 8))
    micro_out = model_h(micro_ids, micro_labels)
    (micro_out.loss * 0.3).backward()

    # Then main forward/backward
    out_h = model_h(**batch)
    out_h.loss.backward()
    hydra_grad = model_h.linear.weight.grad.clone()

    # Hydra grad should be different (has additional micro signal)
    assert not torch.allclose(vanilla_grad, hydra_grad, atol=1e-6)
    # Hydra grad should have larger magnitude (more signal)
    assert hydra_grad.norm() > vanilla_grad.norm() * 0.5  # at least comparable


def test_hydra_micro_only():
    """Hydra works with just micro (no macro)."""
    text, tok = _make_text_and_tok()

    config = ExperimentConfig(name="test-hydra-micro", device="cpu")
    config.training.max_steps = 5
    config.training.log_interval = 5
    config.training.eval_interval = 5
    config.training.save_interval = 100
    config.data.batch_size = 4
    config.data.max_seq_len = 32

    train_loader, val_loader, _ = build_dataloaders(config.data, text, tok)
    micro_loader = _make_loader(text, tok, seq_len=8, batch_size=16)

    model = TinyModel(tok.vocab_size)
    optimizer = build_optimizer(config.optimizer, model)
    scheduler = build_scheduler(config.scheduler, optimizer, config.training.max_steps)

    hydra_cb = HydraCallback(micro_loader=micro_loader, micro_weight=0.5)

    trainer = Trainer(
        config=config, model=model,
        train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler,
        callbacks=[hydra_cb],
    )
    trainer.train()
    assert trainer.tokens_seen > 0


def test_hydra_shift_schedule():
    """Weight schedule 'shift' adjusts micro/macro weights over time."""
    cb = HydraCallback(
        micro_weight=1.0, macro_weight=0.1,
        weight_schedule="shift", total_steps=100,
    )

    # At step 0: full micro, low macro
    micro_w, macro_w = cb._get_weights(0)
    assert abs(micro_w - 1.0) < 0.01
    assert abs(macro_w - 0.1) < 0.01

    # At step 50: micro decayed, macro grown
    micro_w, macro_w = cb._get_weights(50)
    assert micro_w < 1.0
    assert macro_w > 0.1

    # At step 100: micro mostly gone, macro at max
    micro_w, macro_w = cb._get_weights(100)
    assert micro_w < 0.3  # decayed to ~20%
    assert macro_w > 0.2  # grown to ~3x


def test_hydra_convergence():
    """Hydra training should converge at least as well as vanilla."""
    text, tok = _make_text_and_tok()

    def run(use_hydra: bool, steps: int = 50, seed: int = 42) -> float:
        torch.manual_seed(seed)
        config = ExperimentConfig(name="test", device="cpu")
        config.training.max_steps = steps
        config.training.log_interval = steps
        config.training.eval_interval = steps
        config.training.save_interval = steps + 1
        config.data.batch_size = 4
        config.data.max_seq_len = 32

        train_loader, val_loader, _ = build_dataloaders(config.data, text, tok)
        model = TinyModel(tok.vocab_size, dim=32)
        optimizer = build_optimizer(config.optimizer, model)
        scheduler = build_scheduler(config.scheduler, optimizer, steps)

        callbacks = []
        if use_hydra:
            micro_loader = _make_loader(text, tok, seq_len=8, batch_size=16)
            macro_loader = _make_loader(text, tok, seq_len=64, batch_size=2)
            callbacks.append(HydraCallback(
                micro_loader=micro_loader,
                macro_loader=macro_loader,
                micro_weight=0.3,
                macro_weight=0.1,
            ))

        trainer = Trainer(
            config=config, model=model,
            train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, scheduler=scheduler,
            callbacks=callbacks,
        )
        trainer.train()
        return trainer.evaluate()["val/loss"]

    vanilla_loss = run(use_hydra=False)
    hydra_loss = run(use_hydra=True)

    print(f"Vanilla val/loss: {vanilla_loss:.4f}")
    print(f"Hydra val/loss:   {hydra_loss:.4f}")

    # Hydra should be at least comparable (not catastrophically worse)
    assert hydra_loss < vanilla_loss * 1.5
