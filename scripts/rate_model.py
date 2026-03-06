"""Rate a model by actually running forward/backward passes and measuring throughput.

Usage:
    # From a preset
    uv run python scripts/rate_model.py --preset transformer-long

    # From an exported model directory
    uv run python scripts/rate_model.py --model-dir models/my-model

    # From a config JSON
    uv run python scripts/rate_model.py --config configs/tiny_stories.json

    # Directly specify a model class + args
    uv run python scripts/rate_model.py --model transformer --d_model 256 --n_heads 8 --n_layers 6

    # Override workload
    uv run python scripts/rate_model.py --model mamba --d_model 128 --n_layers 7 --batch-size 64 --seq-len 512 --vocab-size 50257
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

from src.config.base import ExperimentConfig, ModelConfig, DataConfig, OptimizerConfig
from src.training.run import _build_model
from src.training.optimizer import build_optimizer, build_scheduler
from src.utils.device import resolve_device


# ── Helpers ──────────────────────────────────────────────

def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _fmt_params(n: int) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.2f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def _fmt_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1e6:.0f} us"
    elif seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    return f"{seconds:.2f} s"


def _fmt_flops(f: float) -> str:
    if f >= 1e12:
        return f"{f / 1e12:.2f} TFLOPS"
    elif f >= 1e9:
        return f"{f / 1e9:.2f} GFLOPS"
    elif f >= 1e6:
        return f"{f / 1e6:.2f} MFLOPS"
    return f"{f:.0f} FLOPS"


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{int(m)}m {int(s)}s"
    else:
        h, rem = divmod(seconds, 3600)
        m = rem / 60
        return f"{int(h)}h {int(m)}m"


def _measure_hw_peak(device: torch.device, size: int = 2048, iters: int = 10) -> float:
    """Quick matmul benchmark to measure actual hardware peak GFLOPS."""
    a = torch.randn(size, size, dtype=torch.float32, device=device)
    b = torch.randn(size, size, dtype=torch.float32, device=device)

    # warmup
    for _ in range(3):
        _ = a @ b
    _sync(device)

    start = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    _sync(device)
    elapsed = (time.perf_counter() - start) / iters

    flops = 2 * size**3 / elapsed
    del a, b
    return flops


# ── Theoretical FLOPS ───────────────────────────────────

def _theoretical_forward_flops(model_type: str, extra: dict, vocab_size: int,
                                n_layers: int, d_model: int, seq_len: int,
                                batch_size: int) -> int:
    tokens = batch_size * seq_len

    if model_type == "transformer":
        attn_qkv = 2 * 3 * d_model * d_model
        attn_scores = 2 * seq_len * d_model
        attn_out = 2 * d_model * d_model
        ffn = 2 * 2 * d_model * 4 * d_model
        per_layer = attn_qkv + attn_scores + attn_out + ffn
        head = 2 * d_model * vocab_size
        return tokens * (n_layers * per_layer + head)

    elif model_type == "mamba":
        expand = extra.get("expand_factor", 2)
        d_inner = d_model * expand
        dt_rank = max(1, d_model // 16)
        d_state = extra.get("d_state", 16)
        d_conv = extra.get("d_conv", 4)
        in_proj = 2 * 2 * d_model * d_inner
        conv = 2 * d_inner * d_conv
        x_proj = 2 * d_inner * (dt_rank + 2 * d_state)
        dt_proj = 2 * dt_rank * d_inner
        ssm = 4 * d_inner * d_state
        out_proj = 2 * d_inner * d_model
        per_layer = in_proj + conv + x_proj + dt_proj + ssm + out_proj
        head = 2 * d_model * vocab_size
        return tokens * (n_layers * per_layer + head)

    elif model_type == "mamba2":
        expand = extra.get("expand_factor", 2)
        d_inner = d_model * expand
        n_heads = extra.get("n_heads", 0)
        if n_heads <= 0:
            n_heads = max(1, d_inner // 64)
        head_dim = d_inner // n_heads
        d_state = extra.get("d_state", 16)
        d_conv = extra.get("d_conv", 4)
        chunk_size = extra.get("chunk_size", 64)
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        # in_proj: d_model -> (d_inner + d_inner + 2*n_heads*d_state + n_heads)
        proj_size = 2 * d_inner + 2 * n_heads * d_state + n_heads
        in_proj = 2 * d_model * proj_size
        # conv1d
        conv = 2 * d_inner * d_conv
        # Intra-chunk: CB matmul per chunk (cs, N) x (cs, N) -> (cs, cs) per head
        intra_cb = num_chunks * n_heads * 2 * chunk_size * chunk_size * d_state
        # Intra-chunk: M @ X per chunk (cs, cs) x (cs, head_dim) -> (cs, head_dim)
        intra_mx = num_chunks * n_heads * 2 * chunk_size * chunk_size * head_dim
        # Inter-chunk: state contrib einsum + state @ C
        inter_contrib = num_chunks * n_heads * 2 * chunk_size * d_state * head_dim
        inter_output = num_chunks * n_heads * 2 * d_state * head_dim
        # out_proj
        out_proj = 2 * d_inner * d_model

        per_layer = in_proj + conv + intra_cb + intra_mx + inter_contrib + inter_output + out_proj
        head_flops = 2 * d_model * vocab_size
        return tokens * (n_layers * per_layer + head_flops)

    elif model_type in ("mamba3", "improved_mamba3"):
        expand = extra.get("expand_factor", 2)
        d_inner = d_model * expand
        n_heads = extra.get("n_heads", 0)
        if n_heads <= 0:
            n_heads = max(1, d_inner // 64)
        head_dim = d_inner // n_heads
        d_state = extra.get("d_state", 16)
        chunk_size = extra.get("chunk_size", 64)
        mlp_factor = extra.get("mlp_factor", 4)
        d_mlp = d_model * mlp_factor
        num_chunks = (seq_len + chunk_size - 1) // chunk_size

        # SSM mixer:
        # in_proj: d_model -> (2*d_inner + 2*d_state + 2*n_heads + d_state//2)
        proj_size = 2 * d_inner + 2 * d_state + 2 * n_heads + d_state // 2
        in_proj = 2 * d_model * proj_size
        # SSD (×2 for trapezoidal two-call decomposition)
        intra_cb = num_chunks * n_heads * 2 * chunk_size * chunk_size * d_state
        intra_mx = num_chunks * n_heads * 2 * chunk_size * chunk_size * head_dim
        inter_contrib = num_chunks * n_heads * 2 * chunk_size * d_state * head_dim
        inter_output = num_chunks * n_heads * 2 * d_state * head_dim
        ssd_one = intra_cb + intra_mx + inter_contrib + inter_output
        ssd_total = 2 * ssd_one  # two SSD calls
        out_proj = 2 * d_inner * d_model

        ssm_per_layer = in_proj + ssd_total + out_proj

        # SwiGLU MLP: gate(d_model→d_mlp) + up(d_model→d_mlp) + down(d_mlp→d_model)
        mlp_per_layer = 3 * 2 * d_model * d_mlp

        per_layer = ssm_per_layer + mlp_per_layer
        head_flops = 2 * d_model * vocab_size
        return tokens * (n_layers * per_layer + head_flops)

    return 0


# ── Source loaders ──────────────────────────────────────

def _config_from_model_dir(model_dir: str) -> tuple[ExperimentConfig, int]:
    config_file = os.path.join(model_dir, "config.json")
    with open(config_file) as f:
        arch = json.load(f)
    model_type = arch.pop("model_type", "transformer")
    vocab_size = arch.pop("vocab_size")
    max_seq_len = arch.pop("max_seq_len", 256)
    return ExperimentConfig(
        model=ModelConfig(name=model_type, extra_args=arch),
        data=DataConfig(max_seq_len=max_seq_len, batch_size=32),
    ), vocab_size


def _config_from_preset(preset: str) -> ExperimentConfig:
    from src.config.presets import get_preset, PRESETS
    cfg = get_preset(preset)
    if cfg is None:
        raise ValueError(f"Unknown preset: {preset}. Available: {list(PRESETS.keys())}")
    return cfg


def _config_from_model_args(model_name: str, extra: dict, batch_size: int,
                             seq_len: int, vocab_size: int) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(name=model_name, extra_args=extra),
        data=DataConfig(
            max_seq_len=seq_len,
            batch_size=batch_size,
            tokenizer_name="char",  # doesn't matter, we pass vocab_size directly
        ),
    )


# ── Main ────────────────────────────────────────────────

def rate_model(
    model_dir: str | None = None,
    config_path: str | None = None,
    preset: str | None = None,
    model_name: str | None = None,
    model_extra: dict | None = None,
    vocab_size: int | None = None,
    batch_size: int | None = None,
    seq_len: int | None = None,
    warmup_steps: int = 5,
    timed_steps: int = 20,
) -> dict:
    """Load the real model, run real fwd/bwd steps, and measure everything."""

    # ── Resolve source ───────────────────────────────────
    _vocab_size = vocab_size
    source_label = ""

    if model_dir is not None:
        config, _vocab_size = _config_from_model_dir(model_dir)
        source_label = model_dir
    elif config_path is not None:
        config = ExperimentConfig.load(config_path)
        source_label = config_path
    elif preset is not None:
        config = _config_from_preset(preset)
        source_label = f"preset:{preset}"
    elif model_name is not None:
        extra = model_extra or {}
        S = seq_len or 256
        B = batch_size or 32
        V = vocab_size or 50257
        config = _config_from_model_args(model_name, extra, B, S, V)
        _vocab_size = V
        source_label = f"{model_name}({', '.join(f'{k}={v}' for k, v in extra.items())})"
    else:
        raise ValueError("Provide one of: model_dir, config_path, preset, or model_name")

    # Apply overrides
    S = seq_len or config.data.max_seq_len
    B = batch_size or config.data.batch_size
    config.data.max_seq_len = S
    config.data.batch_size = B

    # Resolve vocab
    if _vocab_size is None:
        tok_name = config.data.tokenizer_name
        if tok_name == "char":
            _vocab_size = 65
        else:
            _vocab_size = 50257

    device = resolve_device(config.device)
    model_type = config.model.name
    extra = config.model.extra_args
    d_model = extra.get("d_model", 128)
    n_layers = extra.get("n_layers", 4)

    # ── Build model + optimizer ──────────────────────────
    print(f"Building {model_type} model...")
    model = _build_model(config, _vocab_size).to(device)
    optimizer = build_optimizer(config.optimizer, model)
    scheduler = build_scheduler(config.scheduler, optimizer, config.training.max_steps)

    param_count = sum(p.numel() for p in model.parameters())
    tokens_per_step = B * S

    print(f"  {_fmt_params(param_count)} params on {device}")

    # ── Measure hardware peak ────────────────────────────
    print(f"  Measuring hardware peak ({device})...")
    hw_peak = _measure_hw_peak(device)
    print(f"  Hardware peak: {_fmt_flops(hw_peak)} (matmul fp32)")

    print(f"  Running {warmup_steps} warmup + {timed_steps} timed steps...\n")

    # ── Run steps ────────────────────────────────────────
    def _step():
        model.train()
        ids = torch.randint(0, _vocab_size, (B, S), device=device)
        labels = torch.randint(0, _vocab_size, (B, S), device=device)

        optimizer.zero_grad(set_to_none=True)
        output = model(input_ids=ids, labels=labels)
        loss = output.loss
        loss.backward()

        if config.optimizer.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimizer.grad_clip_norm)

        optimizer.step()
        scheduler.step()
        return loss.item()

    # Warmup
    for _ in range(warmup_steps):
        _step()
    _sync(device)

    # Timed
    step_times = []
    for _ in range(timed_steps):
        _sync(device)
        t0 = time.perf_counter()
        _step()
        _sync(device)
        step_times.append(time.perf_counter() - t0)

    # ── Compute stats ────────────────────────────────────
    avg_time = sum(step_times) / len(step_times)
    min_time = min(step_times)
    max_time = max(step_times)
    median_time = sorted(step_times)[len(step_times) // 2]
    stddev = torch.tensor(step_times).std().item()

    tokens_sec = tokens_per_step / avg_time
    tokens_sec_peak = tokens_per_step / min_time

    # Theoretical FLOPS
    theoretical_fwd = _theoretical_forward_flops(
        model_type, extra, _vocab_size, n_layers, d_model, S, B
    )
    theoretical_step = 3 * theoretical_fwd  # fwd + 2x bwd

    achieved_flops = theoretical_step / avg_time
    achieved_flops_peak = theoretical_step / min_time

    # 6N*T
    approx_6nt = 6 * param_count * tokens_per_step

    # MFU against measured hardware peak
    mfu = achieved_flops / hw_peak * 100
    mfu_peak = achieved_flops_peak / hw_peak * 100

    # ── Print report ─────────────────────────────────────
    print("=" * 60)
    print("MODEL BENCHMARK RESULTS")
    print("=" * 60)

    print(f"\nSource: {source_label}")
    print(f"Architecture: {model_type}")
    if model_type == "transformer":
        print(f"  {n_layers}L / {d_model}d / {extra.get('n_heads', 4)}h / vocab={_vocab_size:,}")
    elif model_type == "mamba":
        print(f"  {n_layers}L / {d_model}d / state={extra.get('d_state', 16)} / expand={extra.get('expand_factor', 2)} / vocab={_vocab_size:,}")
    print(f"Parameters: {param_count:,} ({_fmt_params(param_count)})")
    print(f"Device: {device}")

    print(f"\n--- Workload ---")
    print(f"Batch size:   {B}")
    print(f"Seq length:   {S}")
    print(f"Tokens/step:  {tokens_per_step:,}")

    print(f"\n--- Measured Step Time ({timed_steps} steps) ---")
    print(f"Average:  {_fmt_time(avg_time)}")
    print(f"Median:   {_fmt_time(median_time)}")
    print(f"Min:      {_fmt_time(min_time)}")
    print(f"Max:      {_fmt_time(max_time)}")
    print(f"Stddev:   {_fmt_time(stddev)}")

    print(f"\n--- Throughput ---")
    print(f"Tokens/sec (avg):   {tokens_sec:,.0f}")
    print(f"Tokens/sec (peak):  {tokens_sec_peak:,.0f}")

    print(f"\n--- FLOPS ---")
    print(f"Theoretical/step:    {_fmt_flops(theoretical_step)}")
    print(f"Achieved (avg):      {_fmt_flops(achieved_flops)}")
    print(f"Achieved (peak):     {_fmt_flops(achieved_flops_peak)}")
    print(f"6N*T approx/step:    {_fmt_flops(approx_6nt)}")

    print(f"\n--- Hardware Utilization ---")
    print(f"HW peak (measured):  {_fmt_flops(hw_peak)}")
    print(f"MFU (avg):           {mfu:.1f}%")
    print(f"MFU (peak):          {mfu_peak:.1f}%")

    # Training time projections
    total_steps = config.training.max_steps
    total_time = avg_time * total_steps
    total_tokens = tokens_per_step * total_steps

    print(f"\n--- Training Projection ({total_steps:,} steps) ---")
    print(f"Total tokens:  {total_tokens:,}")
    print(f"Estimated:     {_fmt_duration(total_time)}")

    print("\n" + "=" * 60)

    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return {
        "source": source_label,
        "model_type": model_type,
        "param_count": param_count,
        "device": str(device),
        "batch_size": B,
        "seq_len": S,
        "tokens_per_step": tokens_per_step,
        "avg_step_time": avg_time,
        "min_step_time": min_time,
        "tokens_sec_avg": tokens_sec,
        "tokens_sec_peak": tokens_sec_peak,
        "achieved_flops_avg": achieved_flops,
        "achieved_flops_peak": achieved_flops_peak,
        "theoretical_flops_per_step": theoretical_step,
        "hw_peak_flops": hw_peak,
        "mfu_pct": mfu,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark a model by running real forward/backward passes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --preset transformer-long
  %(prog)s --model-dir models/my-model
  %(prog)s --config configs/tiny_stories.json
  %(prog)s --model transformer --d_model 256 --n_heads 8 --n_layers 6
  %(prog)s --model mamba --d_model 128 --n_layers 7 --d_state 16 --expand_factor 2
  %(prog)s --model transformer --d_model 512 --n_layers 12 --vocab-size 50257 --batch-size 64 --seq-len 512
        """,
    )

    # Source (one of these is required)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-dir", type=str, help="Path to exported model in models/")
    source.add_argument("--config", type=str, help="Path to experiment config JSON")
    source.add_argument("--preset", type=str, help="Name of a config preset")
    source.add_argument("--model", type=str, metavar="NAME",
                        help="Model class name: 'transformer' or 'mamba'")

    # Model architecture args (used with --model)
    arch_group = parser.add_argument_group("model architecture (used with --model)")
    arch_group.add_argument("--d_model", type=int, default=128)
    arch_group.add_argument("--n_heads", type=int, default=4)
    arch_group.add_argument("--n_layers", type=int, default=4)
    arch_group.add_argument("--d_state", type=int, default=16, help="Mamba state dim")
    arch_group.add_argument("--d_conv", type=int, default=4, help="Mamba conv width")
    arch_group.add_argument("--expand_factor", type=int, default=2, help="Mamba expand factor")
    arch_group.add_argument("--dropout", type=float, default=0.1)
    arch_group.add_argument("--vocab-size", type=int, default=None,
                            help="Vocab size (default: 65 for char, 50257 for HF)")

    # Workload overrides
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--seq-len", type=int, default=None, help="Override sequence length")

    # Benchmark params
    parser.add_argument("--warmup", type=int, default=5, help="Warmup steps (default: 5)")
    parser.add_argument("--steps", type=int, default=20, help="Timed steps (default: 20)")

    args = parser.parse_args()

    # Build model extra_args from CLI if --model is used
    model_extra = None
    if args.model is not None:
        if args.model == "transformer":
            model_extra = {
                "d_model": args.d_model,
                "n_heads": args.n_heads,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
            }
        elif args.model == "mamba":
            model_extra = {
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "d_state": args.d_state,
                "d_conv": args.d_conv,
                "expand_factor": args.expand_factor,
                "dropout": args.dropout,
            }
        elif args.model == "mamba2":
            model_extra = {
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "d_state": args.d_state,
                "d_conv": args.d_conv,
                "expand_factor": args.expand_factor,
                "chunk_size": 64,
                "dropout": args.dropout,
            }
        elif args.model == "mamba3":
            model_extra = {
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "d_state": args.d_state,
                "expand_factor": args.expand_factor,
                "chunk_size": 64,
                "mlp_factor": 4,
                "dropout": args.dropout,
            }
        elif args.model == "improved_mamba3":
            model_extra = {
                "d_model": args.d_model,
                "n_layers": args.n_layers,
                "n_heads": args.n_heads,
                "d_state": args.d_state,
                "expand_factor": args.expand_factor,
                "chunk_size": 64,
                "mlp_factor": 4,
                "dropout": args.dropout,
            }
        else:
            parser.error(f"Unknown model: {args.model}. Available: transformer, mamba, mamba2, mamba3, improved_mamba3")

    rate_model(
        model_dir=args.model_dir,
        config_path=args.config,
        preset=args.preset,
        model_name=args.model,
        model_extra=model_extra,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        warmup_steps=args.warmup,
        timed_steps=args.steps,
    )


if __name__ == "__main__":
    main()
