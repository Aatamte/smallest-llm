"""Reproduce gradient interference findings from findings/gradient_interference.md.

Measures:
  1. Cross-architecture universality (transformer vs SSM)
  2. Gradient efficiency ratio (signal lost to averaging)
  3. Efficiency scaling with sequence length

Usage: uv run python findings/scripts/gradient_interference.py
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.models import build_model

SEED = 42
V = 256
SAMPLE_TOKENS = 16  # tokens to sample for per-token gradient norms


def gradient_efficiency(model, B: int, L: int) -> dict:
    """Measure gradient efficiency for a model at given batch/seq size."""
    x = torch.randint(0, V, (B, L))
    labels = torch.randint(0, V, (B, L))

    out = model(x, labels=labels)
    per_token_loss = F.cross_entropy(
        out.logits.reshape(-1, V), labels.reshape(-1), reduction="none"
    )
    n_tokens = B * L

    # Full averaged gradient
    model.zero_grad()
    per_token_loss.mean().backward(retain_graph=True)
    full_grad = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    full_norm = full_grad.norm().item()

    # Per-token gradient norms (sampled)
    sample = min(SAMPLE_TOKENS, n_tokens)
    indices = torch.randperm(n_tokens)[:sample]
    norms = []
    for idx in indices:
        model.zero_grad()
        per_token_loss[idx].backward(retain_graph=True)
        g = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        norms.append(g.norm().item())

    avg_norm = sum(norms) / len(norms)
    efficiency = full_norm / (avg_norm + 1e-10)

    return {
        "full_norm": full_norm,
        "avg_per_token_norm": avg_norm,
        "efficiency": efficiency,
        "theory": 1.0 / math.sqrt(n_tokens),
        "waste_pct": (1 - efficiency) * 100,
    }


def position_and_batch_cosine(model, B: int, L: int) -> dict:
    """Measure early-vs-late and batch-vs-batch gradient cosine."""
    quarter = L // 4

    x = torch.randint(0, V, (B, L))
    labels = torch.randint(0, V, (B, L))
    out = model(x, labels=labels)
    per_token_loss = F.cross_entropy(
        out.logits.reshape(-1, V), labels.reshape(-1), reduction="none"
    ).reshape(B, L)

    # Early vs late
    model.zero_grad()
    early_mask = torch.zeros(B, L)
    early_mask[:, :quarter] = 1.0
    ((per_token_loss * early_mask).sum() / early_mask.sum()).backward(retain_graph=True)
    early = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    model.zero_grad()
    late_mask = torch.zeros(B, L)
    late_mask[:, -quarter:] = 1.0
    ((per_token_loss * late_mask).sum() / late_mask.sum()).backward()
    late = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

    pos_cos = F.cosine_similarity(early.unsqueeze(0), late.unsqueeze(0)).item()

    # Batch consistency
    rng = torch.Generator().manual_seed(SEED)
    batch_grads = []
    for _ in range(10):
        model.zero_grad()
        bx = torch.randint(0, V, (B, L), generator=rng)
        bl = torch.randint(0, V, (B, L), generator=rng)
        o = model(bx, labels=bl)
        o.loss.backward()
        batch_grads.append(
            torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        )

    cosines = []
    for i in range(len(batch_grads)):
        for j in range(i + 1, len(batch_grads)):
            cosines.append(
                F.cosine_similarity(batch_grads[i].unsqueeze(0), batch_grads[j].unsqueeze(0)).item()
            )

    return {
        "pos_cosine": pos_cos,
        "batch_cosine": sum(cosines) / len(cosines),
    }


def run():
    B = 2

    # ── 1. Cross-architecture ──
    print("=" * 60)
    print("1. CROSS-ARCHITECTURE UNIVERSALITY")
    print("=" * 60)

    archs = [
        ("transformer", {"d_model": 128, "n_heads": 4, "n_layers": 4, "dropout": 0.1}),
        ("improved_mamba3", {"d_model": 128, "n_layers": 4, "d_state": 16,
                             "expand_factor": 2, "chunk_size": 64, "mlp_factor": 4}),
    ]

    print(f"\n{'Arch':<20s} {'Params':>10s} {'Pos Cosine':>12s} {'Batch Cosine':>14s}")
    print("-" * 58)
    for name, extra in archs:
        model = build_model(name, vocab_size=V, max_seq_len=64, extra_args=extra)
        n_params = sum(p.numel() for p in model.parameters())
        r = position_and_batch_cosine(model, B, 64)
        print(f"{name:<20s} {n_params:>10,d} {r['pos_cosine']:>12.4f} {r['batch_cosine']:>14.4f}")

    # ── 2. Gradient efficiency ──
    print(f"\n{'=' * 60}")
    print("2. GRADIENT EFFICIENCY (signal lost to averaging)")
    print("=" * 60)

    for name, extra in archs:
        model = build_model(name, vocab_size=V, max_seq_len=64, extra_args=extra)
        r = gradient_efficiency(model, B, 64)
        print(f"\n{name}:")
        print(f"  Averaged gradient norm:    {r['full_norm']:.4f}")
        print(f"  Avg per-token grad norm:   {r['avg_per_token_norm']:.4f}")
        print(f"  Efficiency:                {r['efficiency']:.4f}")
        print(f"  Waste:                     {r['waste_pct']:.1f}%")

    # ── 3. Scaling with sequence length ──
    print(f"\n{'=' * 60}")
    print("3. EFFICIENCY vs SEQUENCE LENGTH")
    print("=" * 60)

    model = build_model("transformer", vocab_size=V, max_seq_len=256,
                         extra_args={"d_model": 128, "n_heads": 4, "n_layers": 4, "dropout": 0.1})

    print(f"\n{'Seq Len':>8s} {'Efficiency':>12s} {'Theory':>10s} {'Waste':>8s}")
    print("-" * 40)
    for L in [8, 16, 32, 64, 128]:
        r = gradient_efficiency(model, B, L)
        print(f"{L:>8d} {r['efficiency']:>12.4f} {r['theory']:>10.4f} {r['waste_pct']:>7.1f}%")


if __name__ == "__main__":
    run()
