"""Capture environment info for reproducibility."""

import platform
import subprocess
from datetime import datetime

import torch


def _run_git(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            ["git"] + cmd, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def get_env_info() -> dict:
    """Capture full environment snapshot."""
    return {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "git_hash": _run_git(["rev-parse", "HEAD"]),
        "git_dirty": _run_git(["status", "--porcelain"]) != "",
        "timestamp": datetime.now().isoformat(),
    }
