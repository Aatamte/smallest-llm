"""Test multi-token prediction vs standard next-token prediction.

Compares:
  1. Standard: next-token prediction only
  2. Multi-token (n=4): predict t+1, t+2, t+3, t+4 with decreasing weights

Measures:
  - Validation loss (next-token) convergence
  - Per-horizon prediction quality
  - Generation speed and quality

Usage: uv run python findings/scripts/multi_token_prediction.py
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from src.data.byte_tokenizer import ByteTokenizer
from src.data.streaming import StreamingTextDataset
from src.models import build_model
from src.training.multi_token import MultiTokenHeads, compute_multi_token_loss, generate_multi

SEED = 42
V = 256
L = 64
B = 16
N_STEPS = 500
EVAL_EVERY = 50
N_EVAL_BATCHES = 20
LR = 1e-3
N_AHEAD = 4
MT_WEIGHTS = [1.0, 0.5, 0.25, 0.125]


def build_data():
    """Build train and val dataloaders."""
    from torch.utils.data import DataLoader

    tok = ByteTokenizer()
    train_ds = StreamingTextDataset(
        hf_path="roneneldan/TinyStories", split="train",
        text_field="text", tokenizer=tok, seq_len=L,
    )
    val_ds = StreamingTextDataset(
        hf_path="roneneldan/TinyStories", split="validation",
        text_field="text", tokenizer=tok, seq_len=L, shuffle_buffer=0,
    )
    train_loader = DataLoader(train_ds, batch_size=B)
    val_loader = DataLoader(val_ds, batch_size=B)
    return train_loader, val_loader, tok


def evaluate(model, val_loader, multi_heads=None):
    """Evaluate next-token val loss and per-horizon losses."""
    model.eval()
    if multi_heads is not None:
        multi_heads.eval()

    total_loss = 0.0
    horizon_losses = {i: 0.0 for i in range(1, N_AHEAD + 1)}
    n = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            output = model(input_ids, labels=labels)
            total_loss += output.loss.item()

            # Per-horizon losses
            if multi_heads is not None and output.hidden_states is not None:
                logits = output.logits
                hidden = output.hidden_states
                # t+1: logits[i] predicts labels[i] (no extra shift needed)
                t1 = F.cross_entropy(
                    logits.reshape(-1, V), labels.reshape(-1)
                ).item()
                horizon_losses[1] += t1

                aux_logits = multi_heads(hidden)
                for i, a in enumerate(aux_logits):
                    offset = i + 1
                    if labels.size(1) <= offset:
                        continue
                    h_loss = F.cross_entropy(
                        a[:, :-offset].reshape(-1, V),
                        labels[:, offset:].reshape(-1),
                    ).item()
                    horizon_losses[i + 2] += h_loss

            n += 1
            if n >= N_EVAL_BATCHES:
                break

    model.train()
    if multi_heads is not None:
        multi_heads.train()

    avg_loss = total_loss / max(n, 1)
    avg_horizons = {k: v / max(n, 1) for k, v in horizon_losses.items()}
    return avg_loss, avg_horizons


def train_variant(name, use_multi_token):
    """Train one variant and return val loss history."""
    torch.manual_seed(SEED)
    train_loader, val_loader, tok = build_data()

    model = build_model("transformer", vocab_size=V, max_seq_len=L,
                        extra_args={"d_model": 128, "n_heads": 4, "n_layers": 4, "dropout": 0.0})

    multi_heads = None
    params = list(model.parameters())
    if use_multi_token:
        multi_heads = MultiTokenHeads(128, V, n_ahead=N_AHEAD)
        params = params + list(multi_heads.parameters())

    optimizer = torch.optim.Adam(params, lr=LR)

    history = []
    train_iter = iter(train_loader)

    for step in range(N_STEPS + 1):
        # Eval
        if step % EVAL_EVERY == 0:
            val_loss, horizons = evaluate(model, val_loader, multi_heads)
            history.append((step, val_loss, horizons))
            h_str = ""
            if horizons and any(v > 0 for v in horizons.values()):
                h_str = " | " + " ".join(f"t+{k}={v:.3f}" for k, v in horizons.items() if v > 0)
            print(f"  [{name}] step {step:>4d}  val_loss={val_loss:.4f}{h_str}")

        if step >= N_STEPS:
            break

        # Train step
        model.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"]
        labels = batch["labels"]

        output = model(input_ids, labels=labels)

        loss = output.loss
        if use_multi_token and output.hidden_states is not None:
            aux_loss, _ = compute_multi_token_loss(
                output.hidden_states, labels,
                multi_heads, MT_WEIGHTS,
            )
            loss = loss + aux_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

    return model, multi_heads, history, tok


def benchmark_generation(model, multi_heads, tok):
    """Compare generation speed: standard vs multi-token."""
    prompt = "Once upon a time"
    prompt_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)
    n_tokens = 200

    # Standard generation
    model.eval()
    t0 = time.perf_counter()
    for _ in range(3):
        _ = model.generate(prompt_ids.clone(), max_new_tokens=n_tokens, temperature=0.8)
    std_time = (time.perf_counter() - t0) / 3

    # Multi-token generation
    if multi_heads is not None:
        multi_heads.eval()
        t0 = time.perf_counter()
        for _ in range(3):
            _ = generate_multi(
                model, multi_heads, prompt_ids.clone(),
                max_new_tokens=n_tokens, confidence_threshold=0.7, temperature=0.8,
            )
        mt_time = (time.perf_counter() - t0) / 3
    else:
        mt_time = None

    return std_time, mt_time


def run():
    print("=" * 60)
    print("MULTI-TOKEN PREDICTION EXPERIMENT")
    print("=" * 60)
    print(f"Model: transformer d=128 h=4 L=4 | Data: TinyStories byte")
    print(f"Seq len: {L} | Batch: {B} | Steps: {N_STEPS} | LR: {LR}")
    print(f"Multi-token: n_ahead={N_AHEAD}, weights={MT_WEIGHTS}")
    print()

    # Train standard
    print("Training STANDARD (next-token only)...")
    std_model, _, std_history, tok = train_variant("standard", use_multi_token=False)

    print()

    # Train multi-token
    print("Training MULTI-TOKEN (n_ahead=4)...")
    mt_model, mt_heads, mt_history, _ = train_variant("multi_token", use_multi_token=True)

    # Results table
    print(f"\n{'='*60}")
    print("RESULTS: Validation Loss (next-token)")
    print(f"{'='*60}")
    print(f"{'Step':>6s} | {'Standard':>10s} | {'Multi-Token':>11s} | {'Delta':>8s}")
    print("-" * 42)
    for (s1, l1, _), (s2, l2, _) in zip(std_history, mt_history):
        delta = l2 - l1
        print(f"{s1:>6d} | {l1:>10.4f} | {l2:>11.4f} | {delta:>+8.4f}")

    # Per-horizon losses at final step
    _, _, mt_final_horizons = mt_history[-1]
    if any(v > 0 for v in mt_final_horizons.values()):
        print(f"\n{'='*60}")
        print("PER-HORIZON LOSSES (multi-token model, final step)")
        print(f"{'='*60}")
        for k, v in mt_final_horizons.items():
            if v > 0:
                print(f"  t+{k}: {v:.4f}")

    # Generation benchmark
    print(f"\n{'='*60}")
    print("GENERATION SPEED")
    print(f"{'='*60}")

    std_time, _ = benchmark_generation(std_model, None, tok)
    _, mt_time = benchmark_generation(mt_model, mt_heads, tok)

    print(f"Standard:    {200/std_time:.0f} tok/s ({std_time:.3f}s for 200 tokens)")
    if mt_time:
        print(f"Multi-token: {200/mt_time:.0f} tok/s ({mt_time:.3f}s for 200 tokens)")
        print(f"Speedup:     {std_time/mt_time:.2f}x")

    # Show sample generation
    print(f"\n{'='*60}")
    print("SAMPLE GENERATION")
    print(f"{'='*60}")
    prompt = "Once upon a time"
    prompt_ids = torch.tensor([tok.encode(prompt)], dtype=torch.long)

    mt_model.eval()
    mt_heads.eval()

    print(f"\nPrompt: {prompt!r}")
    std_out = std_model.generate(prompt_ids.clone(), max_new_tokens=100, temperature=0.8)
    print(f"\nStandard: {tok.decode(std_out[0].tolist())!r}")

    mt_out = generate_multi(mt_model, mt_heads, prompt_ids.clone(),
                           max_new_tokens=100, confidence_threshold=0.7, temperature=0.8)
    print(f"\nMulti-token: {tok.decode(mt_out[0].tolist())!r}")


if __name__ == "__main__":
    run()
