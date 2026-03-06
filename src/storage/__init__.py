"""Persistent storage for runs, metrics, checkpoints, models, and eval results."""

from src.storage.impl.main import MainDatabase
from src.storage.impl.eval import EvalDatabase

# Backwards-compatible alias
Database = MainDatabase

__all__ = ["Database", "MainDatabase", "EvalDatabase"]
