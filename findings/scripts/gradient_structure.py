"""Reproduce gradient structure analysis from findings/gradient_structure.md.

Measures:
  1. Information compression (logits → loss → gradient)
  2. Per-group gradient SNR
  3. Batch-to-batch gradient consistency
  4. Hard vs easy token gradient divergence
  5. Early vs late position gradient divergence
  6. A_log gradient by position

Usage: uv run python findings/scripts/gradient_structure.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models import build_model

SEED = 42
B, L, V = 4, 64, 256
N_BATCHES = 10
MODEL_ARGS = {
    "d_model": 128, "n_layers": 4, "d_state": 16,
    "expand_factor": 2, "chunk_size": 64, "mlp_factor": 4,
}


def classify_param(name: str) -> str:
    if "mixer" in name and ("A_log" in name or ".D" in name or "dt_bias" in name):
        return "ssm_dynamics"
    elif "mixer" in name and "proj" in name:
        return "ssm_projections"
    elif "mixer" in name and ("norm" in name or "bias" in name):
        return "ssm_norms"
    elif "mlp" in name:
        return "mlp"
    elif "norm" in name:
        return "layer_norms"
    elif "token_emb" in name or "head" in name:
        return "embeddings"
    return "other"


def classify_param_by_layer(name: str) -> str | None:
    for i in range(10):
        if f"layers.{i}" in name:
            return f"layer_{i}"
    if "token_emb" in name:
        return "embeddings"
    return None


def make_batch(rng: torch.Generator):
    x = torch.randint(0, V, (B, L), generator=rng)
    labels = torch.randint(0, V, (B, L), generator=rng)
    return x, labels


def collect_grads(model) -> dict[str, torch.Tensor]:
    return {
        name: p.grad.detach().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }


def group_grads(grads: dict[str, torch.Tensor], classifier) -> dict[str, torch.Tensor]:
    groups: dict[str, list[torch.Tensor]] = {}
    for name, g in grads.items():
        group = classifier(name)
        if group is None:
            continue
        groups.setdefault(group, []).append(g.flatten())
    return {k: torch.cat(v) for k, v in groups.items()}


def masked_loss(per_token_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (per_token_loss * mask).sum() / mask.sum()


def run():
    rng = torch.Generator().manual_seed(SEED)
    model = build_model("improved_mamba3", vocab_size=V, max_seq_len=L, extra_args=MODEL_ARGS)

    # ── 1. Information compression ──
    print("=" * 60)
    print("1. INFORMATION COMPRESSION")
    print("=" * 60)
    x, labels = make_batch(rng)
    out = model(x, labels=labels)
    n_logits = B * L * V
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Logits: {n_logits:,} values")
    print(f"Loss: 1 value (compression {n_logits:,}:1)")
    print(f"Gradient: {n_params:,} values")

    # ── 2. Per-group gradient SNR ──
    print(f"\n{'=' * 60}")
    print("2. PER-GROUP GRADIENT SNR")
    print("=" * 60)
    model.zero_grad()
    out = model(x, labels=labels)
    out.loss.backward()

    grads = collect_grads(model)
    grouped = group_grads(grads, classify_param)

    print(f"{'Group':<20s} {'#Params':>10s} {'Grad L2':>10s} {'|Mean|/Std':>10s}")
    print("-" * 52)
    for group in sorted(grouped):
        g = grouped[group]
        snr = abs(g.mean().item()) / (g.std().item() + 1e-10)
        print(f"{group:<20s} {g.numel():>10,d} {g.norm().item():>10.4f} {snr:>10.4f}")

    # ── 3. Batch-to-batch consistency ──
    print(f"\n{'=' * 60}")
    print("3. BATCH-TO-BATCH GRADIENT CONSISTENCY")
    print("=" * 60)

    batch_grouped: dict[str, list[torch.Tensor]] = {}
    for _ in range(N_BATCHES):
        model.zero_grad()
        x, labels = make_batch(rng)
        out = model(x, labels=labels)
        out.loss.backward()
        grads = collect_grads(model)
        grouped = group_grads(grads, classify_param)
        for group, g in grouped.items():
            batch_grouped.setdefault(group, []).append(g)

    print(f"{'Group':<20s} {'Avg Cosine':>12s} {'Min':>8s} {'Max':>8s}")
    print("-" * 50)
    for group in sorted(batch_grouped):
        gs = batch_grouped[group]
        cosines = []
        for i in range(len(gs)):
            for j in range(i + 1, len(gs)):
                cos = F.cosine_similarity(gs[i].unsqueeze(0), gs[j].unsqueeze(0)).item()
                cosines.append(cos)
        avg = sum(cosines) / len(cosines)
        print(f"{group:<20s} {avg:>12.4f} {min(cosines):>8.4f} {max(cosines):>8.4f}")

    # ── 4. Hard vs easy token divergence ──
    print(f"\n{'=' * 60}")
    print("4. HARD vs EASY TOKEN GRADIENT DIVERGENCE")
    print("=" * 60)

    model.zero_grad()
    x, labels = make_batch(rng)
    out = model(x, labels=labels)
    logits = out.logits
    per_token_loss = F.cross_entropy(
        logits.reshape(-1, V), labels.reshape(-1), reduction="none"
    ).reshape(B, L)

    sorted_losses, _ = per_token_loss.flatten().sort(descending=True)
    threshold_hard = sorted_losses[len(sorted_losses) // 4].item()
    threshold_easy = sorted_losses[3 * len(sorted_losses) // 4].item()

    # Hard tokens
    model.zero_grad()
    hard_mask = (per_token_loss > threshold_hard).float()
    masked_loss(per_token_loss, hard_mask).backward(retain_graph=True)
    hard_grouped = group_grads(collect_grads(model), classify_param)

    # Easy tokens
    model.zero_grad()
    easy_mask = (per_token_loss < threshold_easy).float()
    masked_loss(per_token_loss, easy_mask).backward()
    easy_grouped = group_grads(collect_grads(model), classify_param)

    print(f"{'Group':<20s} {'Cosine':>10s} {'Hard L2':>10s} {'Easy L2':>10s}")
    print("-" * 52)
    for group in sorted(set(hard_grouped) & set(easy_grouped)):
        h, e = hard_grouped[group], easy_grouped[group]
        cos = F.cosine_similarity(h.unsqueeze(0), e.unsqueeze(0)).item()
        print(f"{group:<20s} {cos:>10.4f} {h.norm().item():>10.4f} {e.norm().item():>10.4f}")

    # ── 5. Early vs late position divergence ──
    print(f"\n{'=' * 60}")
    print("5. EARLY vs LATE POSITION GRADIENT DIVERGENCE")
    print("=" * 60)

    quarter = L // 4
    model.zero_grad()
    x, labels = make_batch(rng)
    out = model(x, labels=labels)
    logits = out.logits
    per_token_loss = F.cross_entropy(
        logits.reshape(-1, V), labels.reshape(-1), reduction="none"
    ).reshape(B, L)

    # Early
    model.zero_grad()
    early_mask = torch.zeros(B, L)
    early_mask[:, :quarter] = 1.0
    masked_loss(per_token_loss, early_mask).backward(retain_graph=True)
    early_grouped = group_grads(collect_grads(model), classify_param_by_layer)

    # Late
    model.zero_grad()
    late_mask = torch.zeros(B, L)
    late_mask[:, -quarter:] = 1.0
    masked_loss(per_token_loss, late_mask).backward()
    late_grouped = group_grads(collect_grads(model), classify_param_by_layer)

    print(f"{'Group':<15s} {'Cosine':>10s} {'Early L2':>10s} {'Late L2':>10s}")
    print("-" * 48)
    for group in sorted(set(early_grouped) & set(late_grouped)):
        e, l = early_grouped[group], late_grouped[group]
        cos = F.cosine_similarity(e.unsqueeze(0), l.unsqueeze(0)).item()
        print(f"{group:<15s} {cos:>10.4f} {e.norm().item():>10.4f} {l.norm().item():>10.4f}")

    # ── 6. A_log gradient by position ──
    print(f"\n{'=' * 60}")
    print("6. A_log GRADIENT BY POSITION")
    print("=" * 60)

    for pos_name, mask_slice in [("EARLY (0-15)", slice(0, quarter)),
                                  ("LATE (48-63)", slice(-quarter, None))]:
        model.zero_grad()
        x, labels = make_batch(rng)
        out = model(x, labels=labels)
        logits = out.logits
        ptl = F.cross_entropy(
            logits.reshape(-1, V), labels.reshape(-1), reduction="none"
        ).reshape(B, L)
        mask = torch.zeros(B, L)
        mask[:, mask_slice] = 1.0
        masked_loss(ptl, mask).backward()

        print(f"\n{pos_name}:")
        for name, p in model.named_parameters():
            if ("A_log" in name or "dt_bias" in name) and p.grad is not None:
                g = p.grad.data
                print(f"  {name:<40s} mean={g.mean():+.6f}  std={g.std():.6f}")


if __name__ == "__main__":
    run()
