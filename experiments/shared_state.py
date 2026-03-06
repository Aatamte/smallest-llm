"""Shared-State Iterative Refinement: order-invariant processing.

Hypothesis: information order shouldn't matter for understanding. Current
architectures (transformers, SSMs) are deeply order-dependent — both in
sequence processing and in layer computation. What if layers communicated
laterally through a shared state instead of a strict pipeline?

Instead of:  x → L1 → L2 → L3 → output  (sequential, order matters)
We do:       state = embed(x)
             repeat N times:
                 state += L1(state) + L2(state) + L3(state)
             output = head(state)

All layers read the same state, all contribute simultaneously, iterate
until the representation stabilizes. Layer order is irrelevant because
they all operate on the same shared state.

This script is self-contained — implements the model, trains it, and
compares against a standard sequential baseline on the same data.

Usage:
    uv run python experiments/shared_state.py
    uv run python experiments/shared_state.py --flops 1e11
    uv run python experiments/shared_state.py --flops 5e12 --d-model 128
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseModel, ModelOutput
from src.models.improved_mamba3 import (
    ImprovedMamba3Layer,
    RMSNorm,
    SwiGLU,
)


# ── Shared-State Model ───────────────────────────────────────────────────────


class SharedStateBlock(nn.Module):
    """A lightweight block for shared-state refinement.

    Unlike ImprovedMamba3Layer which has internal residuals (x + mixer + mlp),
    this block produces a PURE DELTA — no residual connection inside.
    The residual lives in the outer shared-state loop.

    norm → SSM mixer → norm → SwiGLU → output (no skip connections)
    """

    def __init__(self, d_model: int, d_inner: int, n_heads: int,
                 d_state: int, chunk_size: int, d_mlp: int):
        super().__init__()
        from src.models.improved_mamba3 import ImprovedMamba3Mixer
        self.mixer_norm = RMSNorm(d_model)
        self.mixer = ImprovedMamba3Mixer(d_model, d_inner, n_heads, d_state, chunk_size)
        self.mlp_norm = RMSNorm(d_model)
        self.mlp = SwiGLU(d_model, d_mlp)

    def forward(self, x):
        # Pure delta — no residual
        h = self.mixer(self.mixer_norm(x))
        h = h + self.mlp(self.mlp_norm(h))
        return h


class SharedStateMamba(BaseModel):
    """Mamba-3 with shared-state iterative refinement.

    All layers read from and write to a shared state. Each iteration,
    every layer reads the SAME state and produces an independent delta.
    Deltas are summed and added to the state. Layer order is irrelevant.

    Key difference from sequential: layers are weight-shared across
    iterations but produce independent contributions each round. Fewer
    unique layers, more iterations = same compute, better refinement.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 1,
        n_heads: int = 0,
        d_state: int = 16,
        expand_factor: int = 2,
        chunk_size: int = 64,
        mlp_factor: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
        # Shared-state params
        n_iterations: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.n_iterations = n_iterations

        d_inner = d_model * expand_factor
        d_mlp = d_model * mlp_factor

        if n_heads <= 0:
            n_heads = max(1, d_inner // 64)

        assert d_state % 2 == 0, "d_state must be even for RoPE pairing"

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        self.drop = nn.Dropout(dropout)

        # Shared layers — produce pure deltas (no internal residuals)
        self.layers = nn.ModuleList([
            SharedStateBlock(d_model, d_inner, n_heads, d_state, chunk_size, d_mlp)
            for _ in range(n_layers)
        ])

        # Per-iteration norm
        self.iter_norms = nn.ModuleList([
            RMSNorm(d_model) for _ in range(n_iterations)
        ])

        self.norm_f = RMSNorm(d_model)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, **kwargs):
        state = self.drop(self.token_emb(input_ids))

        for t in range(self.n_iterations):
            normed = self.iter_norms[t](state)
            # All layers read the SAME normed state, produce independent deltas
            delta = sum(layer(normed) for layer in self.layers)
            state = state + delta

        x = self.norm_f(state)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return ModelOutput(loss=loss, logits=logits, hidden_states=x)


# ── Sequential baseline (standard Mamba-3 with same param count) ─────────


class SequentialMamba(BaseModel):
    """Standard sequential Mamba-3 for comparison.

    Same layers, same params, but processed in strict order:
    x → L1 → L2 → L3 → output
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 0,
        d_state: int = 16,
        expand_factor: int = 2,
        chunk_size: int = 64,
        mlp_factor: int = 4,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        d_inner = d_model * expand_factor
        d_mlp = d_model * mlp_factor

        if n_heads <= 0:
            n_heads = max(1, d_inner // 64)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            ImprovedMamba3Layer(d_model, d_inner, n_heads, d_state, chunk_size, d_mlp)
            for _ in range(n_layers)
        ])

        self.norm_f = RMSNorm(d_model)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.drop(self.token_emb(input_ids))
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.head(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return ModelOutput(loss=loss, logits=logits, hidden_states=x)


# ── Training loop ────────────────────────────────────────────────────────


def train_model(model, train_loader, val_loader, optimizer, device,
                flops_budget: float, seq_len: int,
                eval_interval_flops: float = 0, label="model"):
    """Training loop bounded by total FLOPs budget.

    Computes flops_per_token from the model, tracks cumulative FLOPs,
    and stops when the budget is exhausted.
    """
    flops_estimate = model.estimate_flops(seq_len)
    flops_per_token = flops_estimate.total

    model.train()
    data_iter = iter(train_loader)
    losses = []
    val_losses = []
    flops_used = 0
    tokens_seen = 0
    step = 0
    last_eval_flops = 0
    t0 = time.perf_counter()

    if eval_interval_flops <= 0:
        eval_interval_flops = flops_budget / 5

    while flops_used < flops_budget:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        batch_tokens = batch["input_ids"].numel()
        tokens_seen += batch_tokens
        flops_used += flops_per_token * batch_tokens
        step += 1
        losses.append(output.loss.item())

        if flops_used - last_eval_flops >= eval_interval_flops:
            val_loss = evaluate(model, val_loader, device)
            val_losses.append((flops_used, val_loss))
            print(f"  [{label}] step={step:>4d} | {flops_used:.2e} FLOPs | "
                  f"train={losses[-1]:.4f} | val={val_loss:.4f}")
            model.train()
            last_eval_flops = flops_used

    wall = time.perf_counter() - t0
    final_val = evaluate(model, val_loader, device)

    return {
        "final_train": sum(losses[-20:]) / min(len(losses), 20),
        "final_val": final_val,
        "wall_time": wall,
        "val_curve": val_losses,
        "flops_used": flops_used,
        "tokens_seen": tokens_seen,
        "steps": step,
        "flops_per_token": flops_per_token,
    }


def evaluate(model, val_loader, device, max_batches=20):
    model.eval()
    total_loss = 0
    n = 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            total_loss += output.loss.item()
            n += 1
    return total_loss / max(n, 1)


def run_eval_suite(model, tokenizer, device):
    """Run ablation_suite eval if available."""
    try:
        from src.evaluation.config import QUICK_EVAL
        from src.evaluation.runner import evaluate as run_eval
        from src.evaluation.trainer_model import TrainerModelWrapper

        wrapper = TrainerModelWrapper(model, tokenizer, device)
        results = run_eval(wrapper, QUICK_EVAL)
        return {name: r.metrics for name, r in results.items()}
    except Exception as e:
        print(f"  Eval suite failed: {e}")
        return None


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Shared-state iterative refinement experiment")
    parser.add_argument("--flops", type=float, default=1e12, help="Total FLOPs budget per config")
    parser.add_argument("--d-model", type=int, default=64, help="Model dimension")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-eval", action="store_true", help="Skip eval suite")
    args = parser.parse_args()

    import gc
    from src.data.datasets import load_dataset, HFStreamingDataset
    from src.data.tokenizer import build_tokenizer
    from src.config.base import DataConfig
    from src.utils.device import resolve_device, empty_cache
    from src.utils.reproducibility import set_seed

    device = resolve_device("auto")
    data_config = DataConfig(
        dataset_name="tiny_stories",
        tokenizer_name="byte",
        max_seq_len=args.seq_len,
        batch_size=args.batch_size,
    )

    dataset = load_dataset("tiny_stories")
    tokenizer = build_tokenizer("byte")

    if isinstance(dataset, HFStreamingDataset):
        from src.data.streaming import build_streaming_dataloaders
        train_loader, val_loader, _ = build_streaming_dataloaders(
            data_config, dataset, tokenizer,
        )
    else:
        from src.data.text import build_dataloaders
        train_loader, val_loader, _ = build_dataloaders(
            data_config, dataset.text, tokenizer,
        )

    # Configs to test:
    # Sequential baseline: 4 unique layers × 1 pass
    # Best shared config from prior runs: 2 layers × 4 iters (best composite)
    configs = [
        ("seq-4L",        "Sequential 4 layers",       {"n_layers": 4}, None),
        ("shared-2L×4",   "Shared 2 layers × 4 iters", {"n_layers": 2}, {"n_iterations": 4}),
    ]

    base_kwargs = dict(
        d_model=args.d_model, d_state=16,
        expand_factor=2, chunk_size=64, mlp_factor=4,
        max_seq_len=args.seq_len, dropout=0.1,
    )

    print("=" * 70)
    print("SHARED-STATE ITERATIVE REFINEMENT EXPERIMENT")
    print(f"FLOPs budget: {args.flops:.2e}, d_model: {args.d_model}")
    print(f"Data: tiny_stories, seq_len: {args.seq_len}, batch: {args.batch_size}")
    print("=" * 70)

    results = {}

    for name, desc, seq_kw, shared_kw in configs:
        print(f"\n{'─' * 70}")
        print(f"{desc}")
        print(f"{'─' * 70}")

        set_seed(args.seed)

        if shared_kw is None:
            # Sequential baseline
            model = SequentialMamba(
                vocab_size=tokenizer.vocab_size, **base_kwargs, **seq_kw,
            ).to(device)
        else:
            model = SharedStateMamba(
                vocab_size=tokenizer.vocab_size, **base_kwargs, **seq_kw, **shared_kw,
            ).to(device)

        params = model.count_parameters()
        flops_est = model.estimate_flops(args.seq_len)
        print(f"  Params: {params:,} | FLOPs/token: {flops_est.total:,} "
              f"(fwd: {flops_est.forward:,}, bwd: {flops_est.backward:,})")

        opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
        r = train_model(
            model, train_loader, val_loader, opt, device,
            flops_budget=args.flops, seq_len=args.seq_len, label=name,
        )
        r["params"] = params

        if not args.no_eval:
            print("  Running eval suite...")
            r["eval"] = run_eval_suite(model, tokenizer, device)

        results[name] = r

        del model, opt
        gc.collect()
        empty_cache(device)

    # ── Report ──
    print(f"\n{'=' * 80}")
    print("RESULTS")
    print(f"{'=' * 80}")

    print(f"\n{'Config':<16} {'Params':>8} {'Val Loss':>10} {'Wall(s)':>8} "
          f"{'Steps':>7} {'FLOPs':>10} {'Tok/FLOP':>10}")
    print("─" * 80)
    for name, r in results.items():
        val_per_flop = r['tokens_seen'] / max(r['flops_used'], 1)
        print(
            f"{name:<16} {r['params']:>8,} {r['final_val']:>10.4f} "
            f"{r['wall_time']:>8.1f} {r['steps']:>7,} {r['flops_used']:>10.2e} "
            f"{val_per_flop:>10.4f}"
        )

    # Eval comparison
    if not args.no_eval and any(r.get("eval") for r in results.values()):
        print(f"\n{'─' * 80}")
        print("TIERED EVAL")
        print(f"{'─' * 80}")
        print(f"{'Config':<16} {'Composite':>9} {'T1:loss':>8} {'T2:phr5':>8} {'T3:cohr':>8} {'T4:avg':>7}")
        print("─" * 60)
        for name, r in results.items():
            if r.get("eval") and "ablation_suite" in r["eval"]:
                m = r["eval"]["ablation_suite"]
                print(
                    f"{name:<16} {m.get('composite', 0):>9.4f} "
                    f"{m.get('t1_loss', 0):>8.3f} {m.get('t2_top5', 0):>8.3f} "
                    f"{m.get('t3_coherence', 0):>8.3f} {m.get('t4_avg', 0):>7.3f}"
                )

    # Compare vs baseline
    baseline = results.get("seq-4L")
    if baseline:
        print(f"\n{'Config':<16} {'Val Δ':>9} {'Wall Δ':>8} {'Param Δ':>8} {'Steps Δ':>8}")
        print("─" * 55)
        for name, r in results.items():
            if name == "seq-4L":
                continue
            val_d = r["final_val"] - baseline["final_val"]
            wall_r = r["wall_time"] / max(baseline["wall_time"], 0.01)
            param_r = r["params"] / baseline["params"]
            step_r = r["steps"] / max(baseline["steps"], 1)
            print(f"{name:<16} {val_d:>+9.4f} {wall_r:>7.2f}x {param_r:>7.2f}x {step_r:>7.2f}x")


if __name__ == "__main__":
    main()
