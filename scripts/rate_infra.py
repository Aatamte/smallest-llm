"""Rate the infrastructure we're working with for LLM training."""

import platform
import time

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_cpu_info():
    info = {
        "processor": platform.processor() or platform.machine(),
        "cores_physical": psutil.cpu_count(logical=False),
        "cores_logical": psutil.cpu_count(logical=True),
        "freq_mhz": None,
    }
    freq = psutil.cpu_freq()
    if freq:
        info["freq_mhz"] = round(freq.max or freq.current)
    return info


def get_memory_info():
    mem = psutil.virtual_memory()
    return {
        "total_gb": round(mem.total / (1024**3), 1),
        "available_gb": round(mem.available / (1024**3), 1),
    }


def get_gpu_info():
    if not torch.cuda.is_available():
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return {"type": "mps", "device": "Apple Silicon GPU"}
        return None

    info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info.append({
            "name": props.name,
            "vram_gb": round(props.total_mem / (1024**3), 1),
            "compute_capability": f"{props.major}.{props.minor}",
        })
    return {"type": "cuda", "devices": info}


def _sync(device):
    """Synchronize the device to get accurate timings."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def _dtype_supported(device, dtype):
    """Check if a dtype is supported on this device."""
    try:
        a = torch.randn(4, 4, dtype=dtype, device=device)
        _ = a @ a
        _sync(device)
        return True
    except (RuntimeError, TypeError):
        return False


def bench_matmul(device, size=2048, dtype=torch.float32, iters=10):
    """Benchmark raw matmul throughput. Returns GFLOPS."""
    a = torch.randn(size, size, dtype=dtype, device=device)
    b = torch.randn(size, size, dtype=dtype, device=device)

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
    return round(flops / 1e9, 2)


def bench_training_step(device, dtype=torch.float32, d_model=256, n_heads=4,
                        n_layers=4, seq_len=128, batch_size=16, vocab_size=4096,
                        iters=10):
    """Benchmark a realistic transformer forward+backward pass.

    Uses torch.autocast for fp16/bf16 (like real mixed-precision training)
    instead of casting the entire model, which avoids dtype mismatches in
    LayerNorm / attention mask.

    Returns dict with tokens/sec, step time, and estimated TFLOPS.
    """
    use_amp = dtype != torch.float32

    class BenchBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
            self.ln2 = nn.LayerNorm(d_model)
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )

        def forward(self, x):
            h = self.ln1(x)
            mask = nn.Transformer.generate_square_subsequent_mask(h.size(1), device=h.device, dtype=h.dtype)
            h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
            x = x + h
            x = x + self.ffn(self.ln2(x))
            return x

    class BenchModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(seq_len, d_model)
            self.blocks = nn.ModuleList([BenchBlock() for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, input_ids, labels):
            T = input_ids.shape[1]
            x = self.tok_emb(input_ids) + self.pos_emb(torch.arange(T, device=input_ids.device))
            for block in self.blocks:
                x = block(x)
            logits = self.head(self.ln_f(x))
            loss = F.cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
            return loss

    # Model stays fp32; autocast handles mixed precision inside forward
    model = BenchModel().to(device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # GradScaler for fp16 on CUDA (not needed for bf16 or MPS)
    use_scaler = use_amp and device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=use_scaler)

    # Determine autocast context
    if use_amp:
        if device.type == "cuda":
            amp_ctx = lambda: torch.autocast("cuda", dtype=dtype)
        elif device.type == "mps":
            # MPS supports autocast since PyTorch 2.3+
            amp_ctx = lambda: torch.autocast("mps", dtype=dtype)
        else:
            amp_ctx = lambda: torch.autocast("cpu", dtype=dtype)
    else:
        from contextlib import nullcontext
        amp_ctx = nullcontext

    tokens_per_step = batch_size * seq_len

    def _step():
        ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        with amp_ctx():
            loss = model(ids, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # warmup
    for _ in range(3):
        _step()
    _sync(device)

    # timed runs
    start = time.perf_counter()
    for _ in range(iters):
        _step()
    _sync(device)
    elapsed = (time.perf_counter() - start) / iters

    tokens_sec = tokens_per_step / elapsed
    param_count = sum(p.numel() for p in model.parameters())

    # Approximate FLOPS: ~6 * params * tokens (forward + backward)
    approx_flops = 6 * param_count * tokens_per_step / elapsed

    del model, optimizer, scaler
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()

    return {
        "step_time_ms": round(elapsed * 1000, 1),
        "tokens_sec": round(tokens_sec),
        "tflops": round(approx_flops / 1e12, 3),
        "param_count": param_count,
    }


def _dtype_label(dtype):
    return {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }.get(dtype, str(dtype))


def rate_infra():
    print("=" * 60)
    print("INFRASTRUCTURE REPORT")
    print("=" * 60)

    # System
    print(f"\nOS: {platform.system()} {platform.release()}")
    print(f"Python: {platform.python_version()}")
    print(f"PyTorch: {torch.__version__}")

    # CPU
    cpu = get_cpu_info()
    print(f"\n--- CPU ---")
    print(f"Processor: {cpu['processor']}")
    print(f"Cores: {cpu['cores_physical']} physical / {cpu['cores_logical']} logical")
    if cpu["freq_mhz"]:
        print(f"Frequency: {cpu['freq_mhz']} MHz")

    # Memory
    mem = get_memory_info()
    print(f"\n--- Memory ---")
    print(f"Total: {mem['total_gb']} GB")
    print(f"Available: {mem['available_gb']} GB")

    # GPU
    gpu = get_gpu_info()
    print(f"\n--- GPU ---")
    if gpu is None:
        print("No GPU detected (CPU-only training)")
    elif gpu["type"] == "mps":
        print(f"Device: {gpu['device']}")
    elif gpu["type"] == "cuda":
        for i, dev in enumerate(gpu["devices"]):
            print(f"GPU {i}: {dev['name']}")
            print(f"  VRAM: {dev['vram_gb']} GB")
            print(f"  Compute: {dev['compute_capability']}")

    # Resolve best device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    # ── Matmul benchmarks across dtypes ──────────────────
    dtypes = [torch.float32, torch.float16, torch.bfloat16]

    print(f"\n--- Matmul Benchmark (2048x2048) ---")
    print(f"{'dtype':<8} {'CPU GFLOPS':>12} {f'GPU ({device.type}) GFLOPS':>20}")
    print("-" * 44)

    best_gflops = 0.0
    best_device_name = "cpu"

    for dtype in dtypes:
        label = _dtype_label(dtype)

        # CPU
        if _dtype_supported(torch.device("cpu"), dtype):
            cpu_gf = bench_matmul(torch.device("cpu"), dtype=dtype)
        else:
            cpu_gf = None

        # GPU
        gpu_gf = None
        if device.type != "cpu" and _dtype_supported(device, dtype):
            gpu_gf = bench_matmul(device, dtype=dtype)

        cpu_str = f"{cpu_gf}" if cpu_gf is not None else "n/a"
        gpu_str = f"{gpu_gf}" if gpu_gf is not None else "n/a"
        print(f"{label:<8} {cpu_str:>12} {gpu_str:>20}")

        for gf, dname in [(cpu_gf, "cpu"), (gpu_gf, device.type)]:
            if gf is not None and gf > best_gflops:
                best_gflops = gf
                best_device_name = dname

    # ── Realistic training benchmark across dtypes ───────
    print(f"\n--- Training Step Benchmark (4L/256d/4h, bs=16, seq=128) ---")
    print(f"{'dtype':<8} {'device':>8} {'step (ms)':>10} {'tok/sec':>10} {'TFLOPS':>10}")
    print("-" * 50)

    best_tok_sec = 0
    best_train_device = "cpu"
    best_train_dtype = "fp32"

    devices_to_bench = [torch.device("cpu")]
    if device.type != "cpu":
        devices_to_bench.append(device)

    for dev in devices_to_bench:
        for dtype in dtypes:
            label = _dtype_label(dtype)
            # Skip half-precision on CPU — autocast doesn't accelerate CPU matmuls
            if dtype != torch.float32 and dev.type == "cpu":
                continue
            if not _dtype_supported(dev, dtype):
                continue

            try:
                result = bench_training_step(dev, dtype=dtype)
                print(
                    f"{label:<8} {dev.type:>8} {result['step_time_ms']:>10} "
                    f"{result['tokens_sec']:>10} {result['tflops']:>10}"
                )

                if result["tokens_sec"] > best_tok_sec:
                    best_tok_sec = result["tokens_sec"]
                    best_train_device = dev.type
                    best_train_dtype = label
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{label:<8} {dev.type:>8}        OOM")
                else:
                    print(f"{label:<8} {dev.type:>8}      error: {e}")

    # ── Rating ───────────────────────────────────────────
    print(f"\n--- Rating ---")
    print(f"Peak matmul: {best_device_name} @ {best_gflops} GFLOPS")
    print(f"Peak training: {best_train_device}/{best_train_dtype} @ {best_tok_sec} tok/sec")

    if best_tok_sec > 100_000:
        tier = "HIGH — can train multi-million param models comfortably"
    elif best_tok_sec > 10_000:
        tier = "MEDIUM — can train small models (< 10M params)"
    elif best_tok_sec > 1_000:
        tier = "LOW — stick to tiny models (< 1M params)"
    else:
        tier = "MINIMAL — character-level / micro models only"

    print(f"Tier: {tier}")
    print(f"Recommended: device={best_train_device}, dtype={best_train_dtype}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    rate_infra()
