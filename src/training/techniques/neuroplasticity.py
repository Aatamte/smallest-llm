"""Neuroplasticity: progressive model growing during training.

The model physically changes size during training:
  - Start tiny (e.g., 1 layer, d_model=32) for fast early learning
  - Grow at scheduled steps (e.g., 2 layers d=64, then 4 layers d=128, etc.)
  - Old weights are preserved; new dimensions get scaled noise

This makes early training 100-1000x faster per step while learning real
features that transfer when the model grows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.training.callbacks import CallbackBase
from src.training.optimizer import build_optimizer, build_scheduler, set_optimizer_lr


@dataclass
class GrowthStage:
    """Defines when and how the model should grow."""
    step: int          # Grow at this training step
    d_model: int
    n_layers: int
    d_state: int = 16
    expand_factor: int = 2
    mlp_factor: int = 4
    n_heads: int = 0   # 0 = auto
    lr: float | None = None  # Optionally reset LR on growth


def _pad_tensor(old: torch.Tensor, new_shape: tuple[int, ...], noise_scale: float) -> torch.Tensor:
    """Copy old tensor into a new larger tensor, fill new dims with scaled noise.

    The old values go into the 'top-left corner' of the new tensor.
    New values are initialized with: randn * (old_std * noise_scale).
    """
    new = torch.randn(new_shape, device=old.device, dtype=old.dtype)
    new *= old.std().item() * noise_scale

    # Build slicing tuple to copy old values into new
    slices = tuple(slice(0, s) for s in old.shape)
    new[slices] = old
    return new


def _pad_1d(old: torch.Tensor, new_size: int, noise_scale: float) -> torch.Tensor:
    return _pad_tensor(old, (new_size,), noise_scale)


def _pad_2d(old: torch.Tensor, new_rows: int, new_cols: int, noise_scale: float) -> torch.Tensor:
    return _pad_tensor(old, (new_rows, new_cols), noise_scale)


def grow_improved_mamba3(
    old_model: nn.Module,
    new_d_model: int,
    new_n_layers: int,
    new_d_state: int = 16,
    new_expand_factor: int = 2,
    new_mlp_factor: int = 4,
    new_n_heads: int = 0,
    new_chunk_size: int = 64,
    noise_scale: float = 0.1,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """Grow a TinyImprovedMamba3 to larger dimensions.

    Creates a new model with the target dimensions and copies weights
    from the old model, padding new dimensions with scaled noise.

    For added layers: new layers are interleaved evenly with existing ones
    to maintain good gradient flow.
    """
    from src.models.improved_mamba3 import TinyImprovedMamba3

    vocab_size = old_model.token_emb.weight.shape[0]
    old_d_model = old_model.d_model
    old_n_layers = len(old_model.layers)

    new_d_inner = new_d_model * new_expand_factor
    new_d_mlp = new_d_model * new_mlp_factor
    if new_n_heads <= 0:
        new_n_heads = max(1, new_d_inner // 64)

    # Create fresh model with target dimensions
    new_model = TinyImprovedMamba3(
        vocab_size=vocab_size,
        d_model=new_d_model,
        n_layers=new_n_layers,
        n_heads=new_n_heads,
        d_state=new_d_state,
        expand_factor=new_expand_factor,
        chunk_size=new_chunk_size,
        mlp_factor=new_mlp_factor,
        max_seq_len=old_model.max_seq_len,
        dropout=old_model.drop.p,
        gradient_checkpointing=gradient_checkpointing,
    )

    device = next(old_model.parameters()).device
    new_model = new_model.to(device)

    with torch.no_grad():
        # ── Token embedding: (vocab_size, d_model) — pad d_model dim ──
        new_model.token_emb.weight[:, :old_d_model] = old_model.token_emb.weight

        # ── Final norm: (d_model,) ──
        new_model.norm_f.weight[:old_d_model] = old_model.norm_f.weight

        # ── Head is weight-tied with token_emb, so it's handled ──

        # ── Layers ──
        # Map old layers to positions in the new model.
        # If we're adding layers, spread old layers evenly.
        if new_n_layers >= old_n_layers:
            # Compute which new layer indices get old weights
            if old_n_layers == 1:
                old_to_new = [0]
            else:
                old_to_new = [
                    round(i * (new_n_layers - 1) / (old_n_layers - 1))
                    for i in range(old_n_layers)
                ]

            for old_idx, new_idx in enumerate(old_to_new):
                _copy_layer_weights(
                    old_model.layers[old_idx],
                    new_model.layers[new_idx],
                    new_d_model, new_d_inner, new_n_heads, new_d_state, new_d_mlp,
                    noise_scale,
                )
        else:
            # Shrinking layers — just copy the first new_n_layers
            for i in range(new_n_layers):
                _copy_layer_weights(
                    old_model.layers[i],
                    new_model.layers[i],
                    new_d_model, new_d_inner, new_n_heads, new_d_state, new_d_mlp,
                    noise_scale,
                )

    return new_model


def _copy_layer_weights(
    old_layer: nn.Module,
    new_layer: nn.Module,
    new_d_model: int,
    new_d_inner: int,
    new_n_heads: int,
    new_d_state: int,
    new_d_mlp: int,
    noise_scale: float,
):
    """Copy weights from an old layer into a new (possibly larger) layer."""
    old_mixer = old_layer.mixer
    new_mixer = new_layer.mixer

    old_d_model = old_mixer.out_proj.weight.shape[0]
    old_d_inner = old_mixer.d_inner
    old_n_heads = old_mixer.n_heads
    old_d_state = old_mixer.d_state

    # ── Mixer norm ──
    new_layer.mixer_norm.weight[:old_d_model] = old_layer.mixer_norm.weight

    # ── in_proj: (d_in_proj_new, d_model_new) ──
    # Split old in_proj output by section, pad each, re-concat
    old_w = old_mixer.in_proj.weight  # (d_in_proj_old, d_model_old)
    old_sections = [
        old_d_inner, old_d_inner,
        old_d_state, old_d_state,
        old_n_heads, old_n_heads,
        old_d_state // 2,
    ]
    new_sections = [
        new_d_inner, new_d_inner,
        new_d_state, new_d_state,
        new_n_heads, new_n_heads,
        new_d_state // 2,
    ]
    old_chunks = old_w.split(old_sections, dim=0)
    new_chunks = []
    for old_chunk, new_out_dim in zip(old_chunks, new_sections):
        padded = _pad_2d(old_chunk, new_out_dim, new_d_model, noise_scale)
        new_chunks.append(padded)
    new_mixer.in_proj.weight.copy_(torch.cat(new_chunks, dim=0))

    # ── A_log, D, dt_bias: (n_heads,) ──
    new_mixer.A_log[:old_n_heads] = old_mixer.A_log
    new_mixer.D[:old_n_heads] = old_mixer.D
    new_mixer.dt_bias[:old_n_heads] = old_mixer.dt_bias

    # ── B_norm, C_norm: (d_state,) ──
    new_mixer.B_norm.weight[:old_d_state] = old_mixer.B_norm.weight
    new_mixer.C_norm.weight[:old_d_state] = old_mixer.C_norm.weight

    # ── B_bias, C_bias: (n_heads, d_state) ──
    new_mixer.B_bias[:old_n_heads, :old_d_state] = old_mixer.B_bias
    new_mixer.C_bias[:old_n_heads, :old_d_state] = old_mixer.C_bias

    # ── out_proj: (d_model, d_inner) ──
    new_mixer.out_proj.weight[:old_d_model, :old_d_inner] = old_mixer.out_proj.weight

    # ── MLP norm ──
    new_layer.mlp_norm.weight[:old_d_model] = old_layer.mlp_norm.weight

    # ── SwiGLU: w_gate_up (d_model, 2*d_mlp) and w_down (d_mlp, d_model) ──
    old_mlp = old_layer.mlp
    new_mlp = new_layer.mlp
    old_d_mlp = old_mlp.w_down.weight.shape[1]  # d_inner of the MLP

    # w_gate_up: (2*d_mlp, d_model) — split into gate and up halves
    old_gu = old_mlp.w_gate_up.weight  # (2*old_d_mlp, old_d_model)
    old_gate, old_up = old_gu.chunk(2, dim=0)
    new_gate = _pad_2d(old_gate, new_d_mlp, new_d_model, noise_scale)
    new_up = _pad_2d(old_up, new_d_mlp, new_d_model, noise_scale)
    new_mlp.w_gate_up.weight.copy_(torch.cat([new_gate, new_up], dim=0))

    # w_down: (d_model, d_mlp)
    new_mlp.w_down.weight[:old_d_model, :old_d_mlp] = old_mlp.w_down.weight


class NeuroplasticityCallback(CallbackBase):
    """Grows the model at scheduled training steps.

    At each growth step:
      1. Creates a new larger model with old weights copied in
      2. Rebuilds the optimizer (new params, fresh momentum)
      3. Rebuilds the scheduler for remaining steps
      4. Optionally resets the learning rate
      5. Grows echo heads if present
    """

    def __init__(self, schedule: list[GrowthStage], remaining_steps_fn=None):
        self.schedule = sorted(schedule, key=lambda s: s.step)
        self._next_idx = 0
        self._remaining_steps_fn = remaining_steps_fn

    def on_step_end(self, trainer, step: int) -> None:
        if self._next_idx >= len(self.schedule):
            return

        stage = self.schedule[self._next_idx]
        if step < stage.step:
            return

        self._next_idx += 1
        print(f"\n=== NEUROPLASTICITY: Growing model at step {step} ===")
        print(f"    d_model={stage.d_model}, n_layers={stage.n_layers}, "
              f"d_state={stage.d_state}, expand={stage.expand_factor}")

        old_params = sum(p.numel() for p in trainer.model.parameters())

        # Grow the model
        new_model = grow_improved_mamba3(
            trainer.model,
            new_d_model=stage.d_model,
            new_n_layers=stage.n_layers,
            new_d_state=stage.d_state,
            new_expand_factor=stage.expand_factor,
            new_mlp_factor=stage.mlp_factor,
            new_n_heads=stage.n_heads,
        )

        new_params = sum(p.numel() for p in new_model.parameters())
        print(f"    {old_params:,} → {new_params:,} params ({new_params/old_params:.1f}x)")

        trainer.model = new_model
        trainer._flops_per_token = trainer._compute_flops_per_token()

        # Grow echo heads if present
        if trainer.echo_heads is not None:
            from src.training.techniques.echo_loss import EchoHeads
            vocab_size = new_model.token_emb.weight.shape[0]
            old_echo = trainer.echo_heads
            new_echo = EchoHeads(stage.d_model, vocab_size).to(trainer.device)

            # Copy old echo head weights where they fit
            old_d = old_echo.backward_head.weight.shape[1]
            with torch.no_grad():
                new_echo.backward_head.weight[:, :old_d] = old_echo.backward_head.weight[:, :old_d]
                for old_sh, new_sh in zip(old_echo.skip_heads, new_echo.skip_heads):
                    new_sh.weight[:, :old_d] = old_sh.weight[:, :old_d]

            trainer.echo_heads = new_echo

        # Rebuild optimizer with new model params
        from src.training.optimizer import build_optimizer
        new_optimizer = build_optimizer(trainer.config.optimizer, trainer.model)
        if stage.lr is not None:
            set_optimizer_lr(new_optimizer, stage.lr)

        # Add echo head params if present
        if trainer.echo_heads is not None:
            new_optimizer.add_param_group({
                "params": list(trainer.echo_heads.parameters()),
                "weight_decay": 0.0,
            })

        trainer.optimizer = new_optimizer

        # Rebuild scheduler for remaining steps
        remaining = trainer.config.training.max_steps - step
        trainer.scheduler = build_scheduler(
            trainer.config.scheduler, trainer.optimizer, remaining,
        )

        print(f"    Optimizer + scheduler rebuilt for {remaining} remaining steps")
