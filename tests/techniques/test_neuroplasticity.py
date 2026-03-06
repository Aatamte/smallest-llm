"""Tests for Neuroplasticity: progressive model growing."""

import torch
import torch.nn.functional as F

from src.config.base import ExperimentConfig
from src.data.text import build_dataloaders
from src.data.tokenizer import build_tokenizer
from src.models.improved_mamba3 import TinyImprovedMamba3
from src.training.techniques.echo_loss import EchoHeads
from src.training.techniques.neuroplasticity import (
    GrowthStage,
    NeuroplasticityCallback,
    grow_improved_mamba3,
)
from src.training.optimizer import build_optimizer, build_scheduler
from src.training.trainer import Trainer


def _make_small_model(vocab_size=64, d_model=32, n_layers=1, d_state=8):
    return TinyImprovedMamba3(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        d_state=d_state,
        expand_factor=2,
        chunk_size=16,
        mlp_factor=4,
        max_seq_len=32,
        dropout=0.0,
    )


def test_grow_preserves_old_weights():
    """After growing, old weights should be in the top-left corner of new weights."""
    old = _make_small_model(d_model=32, n_layers=1, d_state=8)
    old_emb = old.token_emb.weight.clone()
    old_norm = old.norm_f.weight.clone()
    old_out_proj = old.layers[0].mixer.out_proj.weight.clone()

    new = grow_improved_mamba3(old, new_d_model=64, new_n_layers=2, new_d_state=16)

    # Embedding: first 32 dims should match
    assert torch.allclose(new.token_emb.weight[:, :32], old_emb)

    # Final norm: first 32 dims should match
    assert torch.allclose(new.norm_f.weight[:32], old_norm)

    # out_proj of first layer: top-left (32, 64) block should match old (32, 64)
    old_d_inner = 32 * 2  # expand_factor=2
    assert torch.allclose(
        new.layers[0].mixer.out_proj.weight[:32, :old_d_inner],
        old_out_proj,
    )


def test_grow_increases_params():
    """Growing should produce a model with more parameters."""
    old = _make_small_model(d_model=32, n_layers=1, d_state=8)
    new = grow_improved_mamba3(old, new_d_model=64, new_n_layers=2, new_d_state=16)

    old_params = sum(p.numel() for p in old.parameters())
    new_params = sum(p.numel() for p in new.parameters())

    assert new_params > old_params
    print(f"Grew: {old_params:,} → {new_params:,} params ({new_params/old_params:.1f}x)")


def test_grow_forward_backward_works():
    """Grown model should produce valid forward/backward passes."""
    old = _make_small_model(d_model=32, n_layers=1, d_state=8)
    new = grow_improved_mamba3(old, new_d_model=64, new_n_layers=2, new_d_state=16)

    batch, seq_len = 2, 16
    input_ids = torch.randint(0, 64, (batch, seq_len))
    labels = torch.randint(0, 64, (batch, seq_len))

    output = new(input_ids, labels=labels)
    assert output.loss is not None
    assert output.logits.shape == (batch, seq_len, 64)  # vocab_size=64

    output.loss.backward()
    for name, p in new.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


def test_grow_loss_doesnt_spike():
    """Loss after growing shouldn't be catastrophically worse than random init."""
    old = _make_small_model(d_model=32, n_layers=1, d_state=8)
    new = grow_improved_mamba3(old, new_d_model=64, new_n_layers=2, new_d_state=16)

    batch, seq_len, vocab_size = 4, 16, 64
    input_ids = torch.randint(0, vocab_size, (batch, seq_len))
    labels = torch.randint(0, vocab_size, (batch, seq_len))

    with torch.no_grad():
        old_out = old(input_ids, labels=labels)
        new_out = new(input_ids, labels=labels)

    # Random init loss for vocab=64 is ~ln(64) ≈ 4.16
    # Grown model should not be much worse than this
    random_loss = torch.log(torch.tensor(float(vocab_size))).item()
    assert new_out.loss.item() < random_loss * 2, (
        f"Grown model loss {new_out.loss.item():.2f} is too high "
        f"(random baseline ~{random_loss:.2f})"
    )


def test_grow_weight_tying():
    """After growing, head.weight should be tied to token_emb.weight."""
    old = _make_small_model()
    new = grow_improved_mamba3(old, new_d_model=64, new_n_layers=2)

    assert new.head.weight.data_ptr() == new.token_emb.weight.data_ptr()


def test_grow_layer_spreading():
    """When adding layers, old layers should be spread evenly in the new model."""
    old = _make_small_model(d_model=32, n_layers=2, d_state=8)
    old_layer0_norm = old.layers[0].mixer_norm.weight.clone()
    old_layer1_norm = old.layers[1].mixer_norm.weight.clone()

    # Grow to 4 layers: old[0]→new[0], old[1]→new[3]
    new = grow_improved_mamba3(old, new_d_model=32, new_n_layers=4, new_d_state=8)

    assert torch.allclose(new.layers[0].mixer_norm.weight[:32], old_layer0_norm)
    assert torch.allclose(new.layers[3].mixer_norm.weight[:32], old_layer1_norm)


def test_grow_same_dims_is_copy():
    """Growing to the same dimensions should produce identical weights."""
    old = _make_small_model(d_model=32, n_layers=1, d_state=8)
    new = grow_improved_mamba3(
        old, new_d_model=32, new_n_layers=1, new_d_state=8,
        new_expand_factor=2, new_mlp_factor=4, noise_scale=0.0,
    )

    for (n1, p1), (n2, p2) in zip(old.named_parameters(), new.named_parameters()):
        assert torch.allclose(p1, p2, atol=1e-6), f"Mismatch at {n1}"


def test_multistep_growth():
    """Model can be grown multiple times sequentially."""
    m = _make_small_model(d_model=32, n_layers=1, d_state=8)
    m = grow_improved_mamba3(m, new_d_model=48, new_n_layers=2, new_d_state=8)
    m = grow_improved_mamba3(m, new_d_model=64, new_n_layers=3, new_d_state=16)

    # Should still work
    input_ids = torch.randint(0, 64, (2, 16))
    labels = torch.randint(0, 64, (2, 16))
    output = m(input_ids, labels=labels)
    assert output.loss is not None
    output.loss.backward()


def test_callback_grows_at_scheduled_step():
    """NeuroplasticityCallback should grow the model at the right step."""
    config = ExperimentConfig(name="test-neuro", device="cpu")
    config.model.name = "improved_mamba3"
    config.model.extra_args = {"d_model": 32, "n_layers": 1, "d_state": 8,
                               "expand_factor": 2, "mlp_factor": 4}
    config.training.max_steps = 20
    config.training.log_interval = 10
    config.training.eval_interval = 10
    config.training.save_interval = 100
    config.data.batch_size = 2
    config.data.max_seq_len = 16

    text = "the quick brown fox jumps over the lazy dog " * 50
    tok = build_tokenizer("char", text)
    train_loader, val_loader, _ = build_dataloaders(config.data, text, tok)

    model = _make_small_model(vocab_size=tok.vocab_size, d_model=32, n_layers=1, d_state=8)
    optimizer = build_optimizer(config.optimizer, model)
    scheduler = build_scheduler(config.scheduler, optimizer, config.training.max_steps)

    schedule = [
        GrowthStage(step=10, d_model=48, n_layers=2, d_state=8),
    ]
    cb = NeuroplasticityCallback(schedule)

    trainer = Trainer(
        config=config, model=model,
        train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler,
        callbacks=[cb],
    )

    old_params = sum(p.numel() for p in trainer.model.parameters())
    trainer.train()
    new_params = sum(p.numel() for p in trainer.model.parameters())

    assert new_params > old_params, "Model should have grown"
    assert trainer.tokens_seen > 0, "Training should have continued after growth"


def test_callback_with_echo_heads():
    """Callback should also grow echo heads when present."""
    config = ExperimentConfig(name="test-neuro-echo", device="cpu")
    config.training.max_steps = 15
    config.training.log_interval = 10
    config.training.eval_interval = 10
    config.training.save_interval = 100
    config.training.echo_loss = True
    config.data.batch_size = 2
    config.data.max_seq_len = 16

    text = "the quick brown fox jumps over the lazy dog " * 50
    tok = build_tokenizer("char", text)
    train_loader, val_loader, _ = build_dataloaders(config.data, text, tok)

    model = _make_small_model(vocab_size=tok.vocab_size, d_model=32, n_layers=1, d_state=8)
    echo_heads = EchoHeads(32, tok.vocab_size)

    optimizer = build_optimizer(config.optimizer, model)
    optimizer.add_param_group({"params": list(echo_heads.parameters()), "weight_decay": 0.0})
    scheduler = build_scheduler(config.scheduler, optimizer, config.training.max_steps)

    schedule = [GrowthStage(step=8, d_model=48, n_layers=2, d_state=8)]
    cb = NeuroplasticityCallback(schedule)

    trainer = Trainer(
        config=config, model=model,
        train_loader=train_loader, val_loader=val_loader,
        optimizer=optimizer, scheduler=scheduler,
        callbacks=[cb],
        echo_heads=echo_heads,
    )
    trainer.train()

    # Echo heads should now have d_model=48
    assert trainer.echo_heads.backward_head.weight.shape[1] == 48
    assert trainer.tokens_seen > 0
