"""Rate the infrastructure we're working with for LLM training."""

import platform
import time

import psutil
import torch


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


def bench_cpu(size=2048, dtype=torch.float32):
    a = torch.randn(size, size, dtype=dtype)
    b = torch.randn(size, size, dtype=dtype)

    # warmup
    _ = a @ b

    start = time.perf_counter()
    for _ in range(5):
        _ = a @ b
    elapsed = (time.perf_counter() - start) / 5

    flops = 2 * size**3 / elapsed
    return round(flops / 1e9, 2)  # GFLOPS


def bench_gpu(device, size=2048, dtype=torch.float32):
    a = torch.randn(size, size, dtype=dtype, device=device)
    b = torch.randn(size, size, dtype=dtype, device=device)

    # warmup
    for _ in range(3):
        _ = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(10):
        _ = a @ b
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
    elapsed = (time.perf_counter() - start) / 10

    flops = 2 * size**3 / elapsed
    return round(flops / 1e9, 2)  # GFLOPS


def rate_infra():
    print("=" * 50)
    print("INFRASTRUCTURE REPORT")
    print("=" * 50)

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

    # Benchmarks
    print(f"\n--- Benchmarks (matmul {2048}x{2048} fp32) ---")

    cpu_gflops = bench_cpu()
    print(f"CPU: {cpu_gflops} GFLOPS")

    device = None
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")

    if device:
        gpu_gflops = bench_gpu(device)
        print(f"GPU ({device.type}): {gpu_gflops} GFLOPS")
        speedup = round(gpu_gflops / cpu_gflops, 1) if cpu_gflops > 0 else 0
        print(f"GPU/CPU speedup: {speedup}x")

    # Rating
    print(f"\n--- Rating ---")
    best_gflops = gpu_gflops if device else cpu_gflops
    best_device = device.type if device else "cpu"

    if best_gflops > 5000:
        tier = "HIGH — can train multi-million param models"
    elif best_gflops > 500:
        tier = "MEDIUM — can train small models (< 10M params)"
    elif best_gflops > 50:
        tier = "LOW — stick to tiny models (< 1M params)"
    else:
        tier = "MINIMAL — character-level / micro models only"

    print(f"Best device: {best_device} @ {best_gflops} GFLOPS")
    print(f"Tier: {tier}")
    print(f"Recommended training device: {best_device}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    rate_infra()
