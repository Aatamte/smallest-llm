"""Tests for Echo Loss: multi-directional prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config.base import ExperimentConfig
from src.data.text import build_dataloaders
from src.data.tokenizer import build_tokenizer
from src.models.base import BaseModel, ModelOutput
from src.training.techniques.echo_loss import EchoHeads, compute_echo_loss
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.trainer import Trainer


class TinyModelWithHidden(BaseModel):
    """Minimal model that returns hidden states."""
    def __init__(self, vocab_size, dim=16):
        super().__init__()
        self.d_model = dim
        self.max_seq_len = 64
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.linear = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.linear(self.token_emb(input_ids))
        logits = self.head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        return ModelOutput(loss=loss, logits=logits, hidden_states=x)


def test_echo_heads_shapes():
    """Echo heads produce correct output shapes."""
    d_model, vocab_size, batch, seq_len = 32, 100, 4, 16
    heads = EchoHeads(d_model, vocab_size)
    hidden = torch.randn(batch, seq_len, d_model)
    out = heads(hidden)

    assert out["backward"].shape == (batch, seq_len, vocab_size)
    assert out["skip_2"].shape == (batch, seq_len, vocab_size)
    assert out["skip_4"].shape == (batch, seq_len, vocab_size)
    assert out["skip_8"].shape == (batch, seq_len, vocab_size)


def test_echo_loss_computes():
    """Echo loss produces a valid scalar with correct metrics."""
    d_model, vocab_size, batch, seq_len = 32, 100, 4, 16
    heads = EchoHeads(d_model, vocab_size)
    hidden = torch.randn(batch, seq_len, d_model)
    labels = torch.randint(0, vocab_size, (batch, seq_len))

    loss, metrics = compute_echo_loss(hidden, labels, heads)

    assert loss.shape == ()  # scalar
    assert loss.item() > 0
    assert "echo/backward_loss" in metrics
    assert "echo/skip_2_loss" in metrics
    assert "echo/skip_4_loss" in metrics
    assert "echo/skip_8_loss" in metrics
    assert "echo/total_loss" in metrics


def test_echo_loss_short_sequence():
    """Echo loss handles sequences shorter than skip distance gracefully."""
    d_model, vocab_size, batch = 32, 100, 4
    heads = EchoHeads(d_model, vocab_size, skip_distances=(2, 4, 8))

    # seq_len=3: skip_4 and skip_8 should be skipped
    hidden = torch.randn(batch, 3, d_model)
    labels = torch.randint(0, vocab_size, (batch, 3))
    loss, metrics = compute_echo_loss(hidden, labels, heads)

    assert loss.item() > 0
    assert "echo/backward_loss" in metrics
    assert "echo/skip_2_loss" in metrics
    assert "echo/skip_4_loss" not in metrics  # seq too short
    assert "echo/skip_8_loss" not in metrics  # seq too short


def test_echo_gradients_flow_to_model():
    """Gradients from echo loss flow back into the model parameters."""
    d_model, vocab_size, batch, seq_len = 32, 100, 4, 16
    model = TinyModelWithHidden(vocab_size, dim=d_model)
    heads = EchoHeads(d_model, vocab_size)

    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    labels = torch.randint(0, vocab_size, (batch, seq_len))

    output = model(input_ids, labels=labels)
    echo_loss, _ = compute_echo_loss(output.hidden_states, labels, heads)

    # Only backprop echo loss (not the main loss)
    echo_loss.backward()

    # Model params should have gradients from echo loss
    assert model.linear.weight.grad is not None
    assert model.linear.weight.grad.abs().sum() > 0

    # Echo head params should also have gradients
    assert heads.backward_head.weight.grad is not None


def test_echo_combined_loss_backward():
    """Combined main + echo loss backprops correctly."""
    d_model, vocab_size, batch, seq_len = 32, 100, 4, 16
    model = TinyModelWithHidden(vocab_size, dim=d_model)
    heads = EchoHeads(d_model, vocab_size)

    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    labels = torch.randint(0, vocab_size, (batch, seq_len))

    output = model(input_ids, labels=labels)
    echo_loss, _ = compute_echo_loss(output.hidden_states, labels, heads)
    total_loss = output.loss + echo_loss

    total_loss.backward()

    # All model params should have gradients
    for name, p in model.named_parameters():
        assert p.grad is not None, f"No gradient for {name}"


def test_trainer_with_echo_loss():
    """Full integration: trainer runs with echo heads enabled."""
    config = ExperimentConfig(name="test-echo", device="cpu")
    config.training.max_steps = 10
    config.training.log_interval = 5
    config.training.eval_interval = 5
    config.training.save_interval = 100
    config.training.echo_loss = True
    config.data.batch_size = 2
    config.data.max_seq_len = 16

    text = "the quick brown fox jumps over the lazy dog " * 50
    tok = build_tokenizer("char", text)
    train_loader, val_loader, _ = build_dataloaders(config.data, text, tok)

    model = TinyModelWithHidden(tok.vocab_size, dim=16)
    echo_heads = EchoHeads(16, tok.vocab_size)

    optimizer = build_optimizer(config.optimizer, model)
    optimizer.add_param_group({"params": list(echo_heads.parameters()), "weight_decay": 0.0})
    scheduler = build_scheduler(config.scheduler, optimizer, config.training.max_steps)

    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        echo_heads=echo_heads,
    )
    trainer.train()
    assert trainer.tokens_seen > 0


def test_echo_loss_improves_training():
    """Echo loss should lead to lower loss at the same step count.

    This is a convergence sanity check — not a strict assertion,
    but echo should generally help.
    """
    text = "the quick brown fox jumps over the lazy dog " * 100
    tok = build_tokenizer("char", text)
    vocab_size = tok.vocab_size

    def run_training(use_echo: bool, steps: int = 50, seed: int = 42) -> float:
        torch.manual_seed(seed)
        config = ExperimentConfig(name="test", device="cpu")
        config.training.max_steps = steps
        config.training.log_interval = steps
        config.training.eval_interval = steps
        config.training.save_interval = steps + 1
        config.data.batch_size = 4
        config.data.max_seq_len = 16

        train_loader, val_loader, _ = build_dataloaders(config.data, text, tok)
        model = TinyModelWithHidden(vocab_size, dim=32)

        echo_heads = None
        if use_echo:
            echo_heads = EchoHeads(32, vocab_size)

        optimizer = build_optimizer(config.optimizer, model)
        if echo_heads is not None:
            optimizer.add_param_group({"params": list(echo_heads.parameters()), "weight_decay": 0.0})
        scheduler = build_scheduler(config.scheduler, optimizer, steps)

        trainer = Trainer(
            config=config, model=model,
            train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, scheduler=scheduler,
            echo_heads=echo_heads,
        )
        trainer.train()

        # Evaluate final val loss
        val_metrics = trainer.evaluate()
        return val_metrics["val/loss"]

    vanilla_loss = run_training(use_echo=False)
    echo_loss = run_training(use_echo=True)

    print(f"Vanilla val/loss: {vanilla_loss:.4f}")
    print(f"Echo val/loss:    {echo_loss:.4f}")

    # Echo should be at least comparable (not catastrophically worse)
    # In practice it should be better, but we use a loose bound for test stability
    assert echo_loss < vanilla_loss * 1.5, (
        f"Echo loss {echo_loss:.4f} is much worse than vanilla {vanilla_loss:.4f}"
    )
