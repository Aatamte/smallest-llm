"""Tests for utility modules."""

import torch

from src.utils.device import resolve_device, synchronize, empty_cache
from src.utils.env import get_env_info
from src.utils.reproducibility import set_seed, get_rng_states, set_rng_states


def test_resolve_device_cpu():
    device = resolve_device("cpu")
    assert device == torch.device("cpu")


def test_resolve_device_auto():
    device = resolve_device("auto")
    assert device.type in ("cpu", "cuda", "mps")


def test_synchronize_cpu():
    # Should be a no-op, just verify it doesn't crash
    synchronize(torch.device("cpu"))


def test_empty_cache_cpu():
    empty_cache(torch.device("cpu"))


def test_set_seed_deterministic():
    set_seed(42)
    a = torch.randn(10)
    set_seed(42)
    b = torch.randn(10)
    assert torch.equal(a, b)


def test_rng_state_roundtrip():
    set_seed(42)
    states = get_rng_states()
    x = torch.randn(5)
    # Restore and regenerate
    set_rng_states(states)
    y = torch.randn(5)
    assert torch.equal(x, y)


def test_env_info():
    info = get_env_info()
    assert "torch_version" in info
    assert "python_version" in info
    assert "timestamp" in info
