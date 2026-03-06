"""Base Database class — connection setup, table registration, lifecycle."""

from __future__ import annotations

import sqlite3
import threading

from src.storage.table import Table


class Database:
    """Base SQLite database with thread-safe access and table registration.

    Subclasses define their tables and path. On init, all registered
    tables are created automatically.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False, timeout=10.0)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._tables: list[Table] = []

    def _register(self, table: Table):
        self._tables.append(table)

    def _create_all(self):
        """Create all registered tables. Call after registering tables in subclass __init__."""
        for t in self._tables:
            t.create()
            t._init_hash()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    @property
    def lock(self) -> threading.Lock:
        return self._lock

    def set_broadcaster(self, broadcaster):
        """Set a broadcaster on all registered tables for CDC emission."""
        for t in self._tables:
            t.set_broadcaster(broadcaster)

    def get_hashes(self) -> dict[str, int]:
        """Return {table_name: content_hash} for all registered tables."""
        return {t.name: t.get_hash() for t in self._tables}

    def get_schemas(self) -> list[dict]:
        """Return serialized schemas for all registered tables."""
        return [t.to_schema_dict() for t in self._tables]

    def get_table(self, name: str) -> Table:
        """Get a registered table by name. Raises KeyError if not found."""
        for t in self._tables:
            if t.name == name:
                return t
        raise KeyError(f'Table "{name}" not found')

    def close(self):
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
