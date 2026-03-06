"""Ablation experiments + eval probes for training techniques.

Runs controlled experiments with the same model/data/seed, toggling one
technique at a time. Runs eval tasks at the end of each experiment.

Usage:
    uv run python scripts/ablation.py                    # Full suite
    uv run python scripts/ablation.py --steps 50         # Quick smoke test
    uv run python scripts/ablation.py --ablations-only   # Skip diagnostics
    uv run python scripts/ablation.py --diagnostics-only # Skip ablations
    uv run python scripts/ablation.py --no-eval          # Skip eval tasks
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from copy import deepcopy
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.config.base import (
    DataConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
)
from src.evaluation.config import QUICK_EVAL
from src.storage import Database
from src.utils.device import resolve_device, empty_cache


# ── Constants ─────────────────────────────────────────────────────────────────

SEED = 42
DB_PATH = "scripts/ablation.db"
RESULTS_PATH = "scripts/ablation_results.json"

# Small model for fast iteration
BASE_MODEL = ModelConfig(
    name="improved_mamba3",
    extra_args={
        "d_model": 64,
        "n_layers": 4,
        "d_state": 16,
        "expand_factor": 2,
        "chunk_size": 64,
        "mlp_factor": 4,
    },
)

BASE_DATA = DataConfig(
    dataset_name="tiny_shakespeare",
    tokenizer_name="char",
    max_seq_len=512,
    batch_size=8,
)


# ── Experiment Definitions ────────────────────────────────────────────────────


@dataclass
class Experiment:
    name: str
    description: str
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    model: ModelConfig | None = None  # Override BASE_MODEL if set
    data: DataConfig | None = None  # Override BASE_DATA if set


def _base_training(max_steps: int) -> TrainingConfig:
    eval_interval = max(10, max_steps // 10)
    if eval_interval >= max_steps:
        eval_interval = max(1, max_steps // 2)
    return TrainingConfig(
        max_steps=max_steps,
        eval_interval=eval_interval,
        log_interval=5,
        save_interval=max_steps + 1,  # don't save checkpoints
        gradient_accumulation_steps=1,
    )


def _base_optimizer() -> OptimizerConfig:
    return OptimizerConfig(name="adamw", lr=3e-4)


def _muon_optimizer() -> OptimizerConfig:
    return OptimizerConfig(name="muon", lr=0.02, beta1=0.95)


def _base_scheduler() -> SchedulerConfig:
    return SchedulerConfig(warmup_steps=20)


def build_experiments(max_steps: int) -> list[Experiment]:
    """Build experiment set: best known baseline vs new techniques.

    Includes transformer baseline for architecture comparison.
    Control: Muon SSM (proven best from prior ablations).
    Test: Multi-scale SSM init, Hybrid SSM-Attention.
    """
    experiments = []

    # Transformer baseline (AdamW — Muon doesn't help transformers as much)
    transformer_model = ModelConfig(
        name="transformer",
        extra_args={
            "d_model": 64, "n_heads": 2, "n_layers": 4,
        },
    )
    experiments.append(Experiment(
        name="transformer",
        description="Transformer baseline (AdamW)",
        training=_base_training(max_steps),
        optimizer=_base_optimizer(),
        scheduler=_base_scheduler(),
        model=transformer_model,
    ))

    # Control: Mamba-3 SSM + Muon (proven winner)
    experiments.append(Experiment(
        name="mamba3+muon",
        description="Improved Mamba-3 SSM + Muon (best known)",
        training=_base_training(max_steps),
        optimizer=_muon_optimizer(),
        scheduler=_base_scheduler(),
    ))

    # Test 1: Multi-scale SSM (forced timescale separation)
    multiscale_model = ModelConfig(
        name="multiscale_mamba3",
        extra_args={
            "d_model": 64, "n_layers": 4, "d_state": 16,
            "expand_factor": 2, "chunk_size": 64, "mlp_factor": 4,
        },
    )
    experiments.append(Experiment(
        name="multiscale+muon",
        description="Multi-scale SSM (fast/slow head init) + Muon",
        training=_base_training(max_steps),
        optimizer=_muon_optimizer(),
        scheduler=_base_scheduler(),
        model=multiscale_model,
    ))

    # Test 2: Hybrid SSM-Attention (local attention in middle layer)
    hybrid_model = ModelConfig(
        name="hybrid_mamba3",
        extra_args={
            "d_model": 64, "n_layers": 4, "d_state": 16,
            "expand_factor": 2, "chunk_size": 64, "mlp_factor": 4,
            "attn_layer_idx": 2, "attn_window": 64,
        },
    )
    experiments.append(Experiment(
        name="hybrid+muon",
        description="Hybrid SSM-Attention (local attn layer 2) + Muon",
        training=_base_training(max_steps),
        optimizer=_muon_optimizer(),
        scheduler=_base_scheduler(),
        model=hybrid_model,
    ))

    return experiments


# ── Run Ablations ─────────────────────────────────────────────────────────────


def _build_eval_callback(run_id: int, eval_interval: int):
    """Build EvalCallback for mid-training and end-of-training eval."""
    from src.training.callbacks import EvalCallback
    from src.storage import EvalDatabase

    eval_db = EvalDatabase("scripts/ablation_eval.db")
    return EvalCallback(
        tasks=QUICK_EVAL.tasks,
        eval_interval=eval_interval,
        max_samples=QUICK_EVAL.max_samples,
        eval_db=eval_db,
        run_id=run_id,
    ), eval_db


def _run_final_eval(config, model, tokenizer, device, run_id):
    """Run eval tasks on the final model state."""
    from src.evaluation.runner import evaluate
    from src.evaluation.trainer_model import TrainerModelWrapper
    from src.storage import EvalDatabase

    eval_db = EvalDatabase("scripts/ablation_eval.db")
    model_wrapper = TrainerModelWrapper(model, tokenizer, device)

    results = evaluate(
        model_wrapper,
        QUICK_EVAL,
        db=eval_db,
        run_id=run_id,
        step=config.training.max_steps,
    )
    eval_db.close()
    return results


def run_ablations(max_steps: int, run_eval: bool = True) -> list[dict]:
    """Run all ablation experiments and return results."""
    from src.training.run import build_trainer
    from src.utils.reproducibility import set_seed

    db = Database(DB_PATH)
    experiments = build_experiments(max_steps)
    results = []

    print("=" * 70)
    print(f"ABLATION SUITE: {len(experiments)} experiments, {max_steps} steps each")
    print(f"Data: tiny_shakespeare, Seed: {SEED}, DB: {DB_PATH}")
    if run_eval:
        print(f"Eval tasks: {', '.join(QUICK_EVAL.tasks)} (max_samples={QUICK_EVAL.max_samples})")
    print("=" * 70)

    for i, exp in enumerate(experiments):
        print(f"\n{'─' * 70}")
        print(f"[{i+1}/{len(experiments)}] {exp.name}: {exp.description}")
        print(f"{'─' * 70}")

        config = ExperimentConfig(
            name=f"ablation-{exp.name}",
            seed=SEED,
            model=deepcopy(exp.model or BASE_MODEL),
            data=deepcopy(exp.data or BASE_DATA),
            training=exp.training,
            optimizer=exp.optimizer,
            scheduler=exp.scheduler,
            logging=LoggingConfig(db_path=DB_PATH, console_interval=50),
        )

        set_seed(SEED)
        t0 = time.perf_counter()

        try:
            trainer, run_id = build_trainer(config=config, db=db)
            trainer.train()
            db.finish_run(run_id, status="completed")
            wall_time = time.perf_counter() - t0

            # Extract training metrics from DB
            model_cfg = exp.model or BASE_MODEL
            result = _extract_results(db, run_id, exp.name, wall_time, max_steps)
            result["arch"] = model_cfg.name
            result["params"] = trainer.model.count_parameters()

            # Run eval tasks on final model
            if run_eval:
                print(f"  Running eval tasks...")
                eval_results = _run_final_eval(
                    config, trainer.model, trainer.tokenizer, trainer.device, run_id,
                )
                result["eval"] = {
                    name: r.metrics for name, r in eval_results.items()
                }

            results.append(result)

            print(f"  Done in {wall_time:.1f}s — final loss={result['final_train_loss']:.4f}, "
                  f"val={result['final_val_loss']:.4f}")
        except Exception as e:
            wall_time = time.perf_counter() - t0
            print(f"  FAILED after {wall_time:.1f}s: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "name": exp.name,
                "description": exp.description,
                "status": "failed",
                "error": str(e),
                "wall_time": wall_time,
            })

        # Clean up between experiments
        gc.collect()
        device = resolve_device("auto")
        empty_cache(device)

    db.close()
    return results


def _extract_results(db: Database, run_id: int, name: str, wall_time: float,
                     max_steps: int) -> dict:
    """Pull metrics from DB for a completed run."""
    train_loss_rows = db.get_metrics(run_id, "train/loss")
    val_loss_rows = db.get_metrics(run_id, "val/loss")
    step_time_rows = db.get_metrics(run_id, "train/step_time")

    final_train = train_loss_rows[-1]["value"] if train_loss_rows else float("nan")
    final_val = val_loss_rows[-1]["value"] if val_loss_rows else float("nan")

    thresholds = [3.5, 3.0, 2.5]
    steps_to = {}
    for thresh in thresholds:
        step = None
        for row in train_loss_rows:
            if row["value"] <= thresh:
                step = row["step"]
                break
        steps_to[f"steps_to_{thresh}"] = step

    if step_time_rows:
        avg_step_time = sum(r["value"] for r in step_time_rows) / len(step_time_rows)
        batch_size = BASE_DATA.batch_size
        seq_len = BASE_DATA.max_seq_len
        tok_per_sec = (batch_size * seq_len) / avg_step_time
    else:
        avg_step_time = 0
        tok_per_sec = 0

    loss_curve = [(r["step"], r["value"]) for r in train_loss_rows]
    val_curve = [(r["step"], r["value"]) for r in val_loss_rows]

    return {
        "name": name,
        "status": "ok",
        "run_id": run_id,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
        "wall_time": wall_time,
        "avg_step_time": avg_step_time,
        "tok_per_sec": tok_per_sec,
        **steps_to,
        "loss_curve": loss_curve,
        "val_curve": val_curve,
    }


# ── Diagnostic Probes ─────────────────────────────────────────────────────────


def run_diagnostics() -> dict:
    """Run diagnostic probes on a fresh baseline model."""
    from src.data.datasets import load_dataset
    from src.data.text import build_dataloaders
    from src.data.tokenizer import build_tokenizer
    from src.models import build_model
    from src.training.optimizer import build_optimizer
    from src.utils.reproducibility import set_seed

    set_seed(SEED)
    device = resolve_device("auto")

    dataset = load_dataset("tiny_shakespeare")
    tokenizer = build_tokenizer("char", text=dataset.text)
    model = build_model(
        "improved_mamba3",
        vocab_size=tokenizer.vocab_size,
        max_seq_len=128,
        extra_args=BASE_MODEL.extra_args,
    ).to(device)
    optimizer = build_optimizer(_base_optimizer(), model)

    train_loader, _, _ = build_dataloaders(BASE_DATA, dataset.text, tokenizer)

    print("\nWarming up model (20 steps)...")
    model.train()
    data_iter = iter(train_loader)
    for _ in range(20):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        batch = {k: v.to(device) for k, v in batch.items()}
        output = model(**batch)
        output.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    print("Running diagnostics...\n")

    results = {}
    results["token_loss"] = _probe_token_loss(model, train_loader, device)
    results["grad_cosine"] = _probe_grad_cosine(model, train_loader, device)
    results["layer_grads"] = _probe_layer_grads(model, train_loader, device)
    results["state_retention"] = _probe_state_retention(model, train_loader, device)

    del model, optimizer
    gc.collect()
    empty_cache(device)

    return results


def _probe_token_loss(model, loader, device) -> dict:
    model.eval()
    all_losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 5:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            logits = output.logits
            per_token = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["labels"].view(-1),
                reduction="none",
            )
            all_losses.append(per_token.cpu())
    all_losses = torch.cat(all_losses)
    return {
        "mean": all_losses.mean().item(),
        "std": all_losses.std().item(),
        "median": all_losses.median().item(),
        "pct_below_1": (all_losses < 1.0).float().mean().item() * 100,
        "pct_below_0_5": (all_losses < 0.5).float().mean().item() * 100,
    }


def _probe_grad_cosine(model, loader, device) -> dict:
    model.train()
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    n_examples = min(8, batch["input_ids"].size(0))
    per_example_grads = []
    for i in range(n_examples):
        model.zero_grad(set_to_none=True)
        single = {"input_ids": batch["input_ids"][i:i+1], "labels": batch["labels"][i:i+1]}
        output = model(**single)
        output.loss.backward()
        grad_vec = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
        per_example_grads.append(grad_vec)
    model.zero_grad(set_to_none=True)

    similarities = []
    for i in range(len(per_example_grads)):
        for j in range(i + 1, len(per_example_grads)):
            cos = F.cosine_similarity(
                per_example_grads[i].unsqueeze(0),
                per_example_grads[j].unsqueeze(0),
            ).item()
            similarities.append(cos)
    return {
        "mean_cosine": sum(similarities) / max(len(similarities), 1),
        "min_cosine": min(similarities) if similarities else 0,
        "max_cosine": max(similarities) if similarities else 0,
        "n_pairs": len(similarities),
    }


def _probe_layer_grads(model, loader, device) -> dict:
    model.train()
    model.zero_grad(set_to_none=True)
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}
    output = model(**batch)
    output.loss.backward()

    layer_norms = []
    for i, layer in enumerate(model.layers):
        norm = sum(p.grad.norm().item() ** 2 for p in layer.parameters() if p.grad is not None) ** 0.5
        layer_norms.append(norm)
    model.zero_grad(set_to_none=True)

    ratio = layer_norms[0] / max(layer_norms[-1], 1e-10) if len(layer_norms) >= 2 else 1.0
    return {"per_layer_norm": layer_norms, "first_last_ratio": ratio}


def _probe_state_retention(model, loader, device) -> dict:
    model.eval()
    batch = next(iter(loader))
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        output = model(**batch)
        hidden = output.hidden_states

    if hidden is None:
        return {"error": "Model does not return hidden_states"}

    vocab_size = model.token_emb.weight.shape[0]
    d_model = hidden.shape[-1]
    seq_len = hidden.shape[1]

    results = {}
    for k in [1, 2, 4, 8, 16]:
        if k >= seq_len:
            break
        h = hidden[:, k:, :].reshape(-1, d_model)
        targets = batch["input_ids"][:, :seq_len - k].reshape(-1)

        probe = torch.nn.Linear(d_model, vocab_size, bias=False).to(device)
        probe_opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        for _ in range(100):
            logits = probe(h)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            probe_opt.step()
            probe_opt.zero_grad(set_to_none=True)

        with torch.no_grad():
            acc = (probe(h).argmax(dim=-1) == targets).float().mean().item()
        results[f"k={k}_acc"] = acc
        del probe, probe_opt

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────


def print_ablation_report(results: list[dict], max_steps: int):
    print("\n" + "=" * 100)
    print(f"ABLATION RESULTS ({max_steps} steps, seed={SEED})")
    print("=" * 100)

    # Main results table
    print(f"\n{'Experiment':<18} {'Arch':<20} {'Params':>8} {'Val@end':>8} {'Wall(s)':>8} {'tok/s':>8}")
    print("─" * 74)
    for r in results:
        if r.get("status") == "failed":
            print(f"{r['name']:<18} {'':>20} {'FAILED':>8}   {r.get('error', '')[:40]}")
            continue
        arch = r.get("arch", "?")
        params = r.get("params", 0)
        print(
            f"{r['name']:<18} "
            f"{arch:<20} "
            f"{params:>8,} "
            f"{r['final_val_loss']:>8.4f} "
            f"{r['wall_time']:>8.1f} "
            f"{r['tok_per_sec']:>8,.0f}"
        )

    # Tiered eval results
    has_eval = any(r.get("eval") for r in results if r.get("status") == "ok")
    if has_eval:
        # Check if any result has ablation_suite metrics (tiered)
        has_tiered = any(
            "ablation_suite" in r.get("eval", {})
            and "composite" in r["eval"].get("ablation_suite", {})
            for r in results if r.get("status") == "ok"
        )

        if has_tiered:
            print(f"\n{'─' * 90}")
            print("TIERED EVAL (end of training)")
            print(f"{'─' * 90}")
            print(
                f"{'Experiment':<18} {'Composite':>9} "
                f"{'T1:loss':>8} {'T1:top1':>8} "
                f"{'T2:phr5':>8} {'T3:cohr':>8} "
                f"{'T4:avg':>8}"
            )
            print("─" * 78)

            for r in results:
                if r.get("status") != "ok" or "ablation_suite" not in r.get("eval", {}):
                    continue
                m = r["eval"]["ablation_suite"]
                print(
                    f"{r['name']:<18} "
                    f"{m.get('composite', 0):>9.4f} "
                    f"{m.get('t1_loss', 0):>8.4f} "
                    f"{m.get('t1_top1', 0):>8.4f} "
                    f"{m.get('t2_top5', 0):>8.4f} "
                    f"{m.get('t3_coherence', 0):>8.4f} "
                    f"{m.get('t4_avg', 0):>8.4f}"
                )

            # State tracking detail
            print(f"\n  State tracking detail:")
            for r in results:
                if r.get("status") != "ok" or "ablation_suite" not in r.get("eval", {}):
                    continue
                m = r["eval"]["ablation_suite"]
                parts = []
                for k in ["t4_counting", "t4_recall", "t4_pattern"]:
                    if k in m:
                        parts.append(f"{k.split('_')[1]}={m[k]:.3f}")
                if parts:
                    print(f"    {r['name']:<16} {', '.join(parts)}")
        else:
            # Fallback: generic eval table (for non-tiered tasks)
            print(f"\n{'─' * 90}")
            print("EVAL RESULTS (end of training)")
            print(f"{'─' * 90}")

            all_keys: list[str] = []
            for r in results:
                if r.get("eval"):
                    for task_name, metrics in r["eval"].items():
                        for k in metrics:
                            full_key = f"{task_name}/{k}"
                            if full_key not in all_keys:
                                all_keys.append(full_key)

            key_labels = [k.split("/")[-1][:12] for k in all_keys]
            header = f"{'Experiment':<18} " + " ".join(f"{l:>12}" for l in key_labels)
            print(header)
            print("─" * len(header))

            for r in results:
                if r.get("status") != "ok" or not r.get("eval"):
                    continue
                vals = []
                for full_key in all_keys:
                    task, metric = full_key.split("/", 1)
                    v = r["eval"].get(task, {}).get(metric)
                    vals.append(f"{v:>12.4f}" if v is not None else f"{'n/a':>12}")
                print(f"{r['name']:<18} " + " ".join(vals))

    # Relative comparison vs control (mamba3+muon is the primary control)
    control = next(
        (r for r in results if r["name"] == "mamba3+muon" and r.get("status") == "ok"),
        next((r for r in results if r.get("status") == "ok"), None),
    )
    if control and len(results) > 1:
        base_val = control["final_val_loss"]
        if not math.isnan(base_val):
            print(f"\n{'Experiment':<18} {'Val Δ':>9} {'Wall Δ':>9} {'Comp Δ':>9} {'Verdict':>10}")
            print("─" * 60)
            for r in results:
                if r.get("status") != "ok" or r["name"] == control["name"]:
                    continue
                val_delta = r["final_val_loss"] - base_val
                wall_ratio = r["wall_time"] / max(control["wall_time"], 0.01)

                # Composite delta if available
                ctrl_comp = control.get("eval", {}).get("ablation_suite", {}).get("composite")
                test_comp = r.get("eval", {}).get("ablation_suite", {}).get("composite")
                if ctrl_comp is not None and test_comp is not None:
                    comp_delta = test_comp - ctrl_comp
                    comp_str = f"{comp_delta:>+9.4f}"
                else:
                    comp_delta = None
                    comp_str = f"{'n/a':>9}"

                if math.isnan(r["final_val_loss"]):
                    verdict = "DIVERGED"
                elif comp_delta is not None and comp_delta > 0.02 and wall_ratio < 1.2:
                    verdict = "WINNER"
                elif val_delta < -0.05 and wall_ratio < 1.2:
                    verdict = "WINNER"
                elif comp_delta is not None and comp_delta > 0.01:
                    verdict = "helpful"
                elif val_delta < -0.02:
                    verdict = "helpful"
                elif val_delta > 0.1:
                    verdict = "HURT"
                elif wall_ratio > 1.3 and val_delta > -0.02:
                    verdict = "too slow"
                else:
                    verdict = "neutral"
                print(f"{r['name']:<18} {val_delta:>+9.4f} {wall_ratio:>8.2f}x {comp_str} {verdict:>10}")


def print_diagnostic_report(diag: dict):
    print("\n" + "=" * 80)
    print("DIAGNOSTIC PROBES")
    print("=" * 80)

    tl = diag.get("token_loss", {})
    if tl:
        print(f"\n1. Per-token loss: mean={tl['mean']:.3f}, std={tl['std']:.3f}, median={tl['median']:.3f}")
        print(f"   {tl['pct_below_1']:.1f}% below 1.0, {tl['pct_below_0_5']:.1f}% below 0.5")

    gc_res = diag.get("grad_cosine", {})
    if gc_res:
        print(f"\n2. Gradient cosine sim: mean={gc_res['mean_cosine']:.4f} "
              f"[{gc_res['min_cosine']:.4f}, {gc_res['max_cosine']:.4f}]")

    lg = diag.get("layer_grads", {})
    if lg:
        print(f"\n3. Layer grad norms: " + ", ".join(f"L{i}={n:.4f}" for i, n in enumerate(lg["per_layer_norm"])))
        print(f"   First/Last ratio: {lg['first_last_ratio']:.2f}")

    sr = diag.get("state_retention", {})
    if sr and "error" not in sr:
        print(f"\n4. State retention: " + ", ".join(
            f"k={k.split('=')[1].split('_')[0]}={v:.3f}"
            for k, v in sorted(sr.items()) if k.endswith("_acc")
        ))


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Ablation experiments")
    parser.add_argument("--steps", type=int, default=500, help="Training steps per experiment")
    parser.add_argument("--ablations-only", action="store_true", help="Skip diagnostics")
    parser.add_argument("--diagnostics-only", action="store_true", help="Skip ablations")
    parser.add_argument("--no-eval", action="store_true", help="Skip eval tasks after training")
    args = parser.parse_args()

    all_results = {}

    if not args.diagnostics_only:
        ablation_results = run_ablations(args.steps, run_eval=not args.no_eval)
        print_ablation_report(ablation_results, args.steps)
        all_results["ablations"] = ablation_results

    if not args.ablations_only:
        diag_results = run_diagnostics()
        print_diagnostic_report(diag_results)
        all_results["diagnostics"] = diag_results

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
