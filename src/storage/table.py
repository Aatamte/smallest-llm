"""Base Table abstraction for declarative SQLite table definitions."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field


@dataclass
class Column:
    name: str
    type: str  # e.g. "INTEGER", "TEXT", "REAL"
    primary_key: bool = False
    autoincrement: bool = False
    not_null: bool = False
    default: str | None = None  # Raw SQL default, e.g. "datetime('now')"
    references: str | None = None  # e.g. "runs(id)"


@dataclass
class Index:
    name: str
    columns: list[str]


class Table:
    """Base class for a SQLite table.

    Subclasses declare `name`, `columns`, and `indexes`, then get
    automatic DDL generation and basic CRUD helpers.
    """

    name: str = ""
    columns: list[Column] = []
    indexes: list[Index] = []

    def __init__(self, conn: sqlite3.Connection, lock: threading.Lock):
        self._conn = conn
        self._lock = lock
        self._hash: int = 0
        self._broadcaster = None  # Optional: set via set_broadcaster()
        self._col_names: list[str] = [c.name for c in self.columns]
        pk_cols = [c for c in self.columns if c.primary_key]
        if len(pk_cols) != 1:
            raise ValueError(
                f'Table "{self.name}" must have exactly one primary key column, got {len(pk_cols)}'
            )
        self.pk_col: str = pk_cols[0].name

    def set_broadcaster(self, broadcaster):
        """Set a broadcaster to emit CDC ops on mutations."""
        self._broadcaster = broadcaster

    def _init_hash(self):
        """Compute hash from existing rows in the table."""
        with self._lock:
            self._rehash_unlocked()

    def create(self):
        """Generate and execute CREATE TABLE + CREATE INDEX statements."""
        col_defs = []
        for c in self.columns:
            parts = [c.name, c.type]
            if c.primary_key:
                parts.append("PRIMARY KEY")
            if c.autoincrement:
                parts.append("AUTOINCREMENT")
            if c.not_null:
                parts.append("NOT NULL")
            if c.default is not None:
                parts.append(f"DEFAULT ({c.default})")
            if c.references:
                parts.append(f"REFERENCES {c.references}")
            col_defs.append(" ".join(parts))

        ddl = f"CREATE TABLE IF NOT EXISTS {self.name} (\n"
        ddl += ",\n".join(f"    {d}" for d in col_defs)
        ddl += "\n);"

        idx_stmts = []
        for idx in self.indexes:
            cols = ", ".join(idx.columns)
            idx_stmts.append(
                f"CREATE INDEX IF NOT EXISTS {idx.name} ON {self.name}({cols});"
            )

        with self._lock:
            self._conn.execute(ddl)
            for stmt in idx_stmts:
                self._conn.execute(stmt)
            self._conn.commit()

    def insert(self, **kwargs) -> int:
        cols = list(kwargs.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_names = ", ".join(cols)
        values = [kwargs[c] for c in cols]
        with self._lock:
            cursor = self._conn.execute(
                f"INSERT INTO {self.name} ({col_names}) VALUES ({placeholders})",
                values,
            )
            self._conn.commit()
            rowid = cursor.lastrowid
            # XOR in the new row's hash
            row = self._get_by_pk_unlocked(rowid)
            if row:
                self._hash ^= self._hash_row(row)
                self._emit_op("upsert", row=row)
            return rowid

    def insert_many(self, rows: list[dict]) -> list[int]:
        """Batch INSERT with one transaction, one read-back, one CDC emit."""
        if not rows:
            return []
        rowids: list[int] = []
        with self._lock:
            for row in rows:
                cols = list(row.keys())
                placeholders = ", ".join("?" for _ in cols)
                col_names = ", ".join(cols)
                values = [row[c] for c in cols]
                cursor = self._conn.execute(
                    f"INSERT INTO {self.name} ({col_names}) VALUES ({placeholders})",
                    values,
                )
                rowids.append(cursor.lastrowid)
            self._conn.commit()
            # Batch read-back and hash
            inserted_rows = []
            for rowid in rowids:
                full_row = self._get_by_pk_unlocked(rowid)
                if full_row:
                    self._hash ^= self._hash_row(full_row)
                    inserted_rows.append(full_row)
            # One batched CDC emit
            if inserted_rows:
                self._emit_ops("upsert", rows=inserted_rows)
        return rowids

    def select(
        self,
        where: str | None = None,
        params: list | None = None,
        order_by: str | None = None,
        columns: str = "*",
    ) -> list[dict]:
        query = f"SELECT {columns} FROM {self.name}"
        if where:
            query += f" WHERE {where}"
        if order_by:
            query += f" ORDER BY {order_by}"
        with self._lock:
            rows = self._conn.execute(query, params or []).fetchall()
            return [dict(r) for r in rows]

    def delete(self, where: str, params: list | None = None):
        with self._lock:
            # XOR out hashes of rows being deleted
            rows = self._conn.execute(
                f"SELECT * FROM {self.name} WHERE {where}", params or []
            ).fetchall()
            for row in rows:
                row_dict = dict(row)
                self._hash ^= self._hash_row(row_dict)
                self._emit_op("delete", key=row_dict[self.pk_col])
            self._conn.execute(f"DELETE FROM {self.name} WHERE {where}", params or [])
            self._conn.commit()

    def update(self, set_clause: str, where: str, params: list | None = None):
        with self._lock:
            all_params = params or []
            # Split params: count ? in set_clause for SET params, rest are WHERE params
            set_count = set_clause.count("?")
            where_params = all_params[set_count:]
            # Read full rows before update to XOR out old hashes
            old_rows = self._conn.execute(
                f"SELECT * FROM {self.name} WHERE {where}", where_params
            ).fetchall()
            for row in old_rows:
                self._hash ^= self._hash_row(dict(row))
            pk_vals = [dict(row)[self.pk_col] for row in old_rows]
            self._conn.execute(
                f"UPDATE {self.name} SET {set_clause} WHERE {where}", all_params
            )
            self._conn.commit()
            # Read back updated rows, XOR in new hashes, emit CDC
            updated_rows = []
            for pk in pk_vals:
                row = self._get_by_pk_unlocked(pk)
                if row:
                    self._hash ^= self._hash_row(row)
                    updated_rows.append(row)
            if updated_rows:
                self._emit_ops("upsert", rows=updated_rows)

    def upsert(self, **kwargs) -> int:
        """INSERT OR REPLACE a row. Requires explicit primary key value."""
        pk_val = kwargs.get(self.pk_col)
        if pk_val is None:
            raise ValueError(f'Row must include primary key column "{self.pk_col}"')
        with self._lock:
            # XOR out old row hash if it exists
            existing = self._get_by_pk_unlocked(pk_val)
            if existing:
                self._hash ^= self._hash_row(existing)
            cols = list(kwargs.keys())
            placeholders = ", ".join("?" for _ in cols)
            col_names = ", ".join(cols)
            values = [kwargs[c] for c in cols]
            self._conn.execute(
                f"INSERT OR REPLACE INTO {self.name} ({col_names}) VALUES ({placeholders})",
                values,
            )
            self._conn.commit()
            # XOR in the new row's hash (read back to get defaults)
            inserted = self._get_by_pk_unlocked(pk_val)
            if inserted:
                self._hash ^= self._hash_row(inserted)
                self._emit_op("upsert", row=inserted)
            return pk_val

    def upsert_many(self, rows: list[dict]):
        """Batch INSERT OR REPLACE. Each row must include the primary key."""
        if not rows:
            return
        for row in rows:
            pk_val = row.get(self.pk_col)
            if pk_val is None:
                raise ValueError(f'Row must include primary key column "{self.pk_col}"')
        with self._lock:
            upserted_rows = []
            for row in rows:
                pk_val = row[self.pk_col]
                existing = self._get_by_pk_unlocked(pk_val)
                if existing:
                    self._hash ^= self._hash_row(existing)
                cols = list(row.keys())
                placeholders = ", ".join("?" for _ in cols)
                col_names = ", ".join(cols)
                values = [row[c] for c in cols]
                self._conn.execute(
                    f"INSERT OR REPLACE INTO {self.name} ({col_names}) VALUES ({placeholders})",
                    values,
                )
                inserted = self._get_by_pk_unlocked(pk_val)
                if inserted:
                    self._hash ^= self._hash_row(inserted)
                    upserted_rows.append(inserted)
            self._conn.commit()
            if upserted_rows:
                self._emit_ops("upsert", rows=upserted_rows)

    def clear(self):
        """Delete all rows from the table."""
        with self._lock:
            self._conn.execute(f"DELETE FROM {self.name}")
            self._conn.commit()
            self._hash = 0
            self._emit_op("clear")

    def get_all_rows(self) -> list[dict]:
        """Return all rows as a list of dicts."""
        return self.select()

    def get_hash(self) -> int:
        """Current content hash (XOR of all row hashes)."""
        return self._hash

    def to_schema_dict(self) -> dict:
        """Serialize table schema to JSON-compatible dict matching frontend TableSchema."""
        columns = []
        for c in self.columns:
            col: dict = {"name": c.name, "type": c.type}
            if c.primary_key:
                col["primaryKey"] = True
            if c.autoincrement:
                col["autoincrement"] = True
            if c.not_null:
                col["notNull"] = True
            if c.default is not None:
                col["default"] = c.default
            columns.append(col)
        schema: dict = {"name": self.name, "columns": columns}
        if self.indexes:
            schema["indexes"] = [
                {"name": idx.name, "columns": idx.columns} for idx in self.indexes
            ]
        return schema

    # ── Private helpers ─────────────────────────────────────

    @staticmethod
    def _djb2(s: str) -> int:
        """djb2 hash over a string → unsigned 32-bit integer."""
        h = 5381
        for ch in s:
            h = ((h << 5) + h + ord(ch)) & 0xFFFFFFFF
        return h

    def _hash_row(self, row: dict) -> int:
        """Hash a row's values using djb2. Must match frontend implementation."""
        sorted_cols = sorted(self._col_names)
        s = json.dumps([row.get(c) for c in sorted_cols], separators=(",", ":"))
        return self._djb2(s)

    def _get_by_pk(self, key) -> dict | None:
        """Look up a single row by primary key. Acquires lock."""
        with self._lock:
            return self._get_by_pk_unlocked(key)

    def _get_by_pk_unlocked(self, key) -> dict | None:
        """Look up a single row by primary key. Caller must hold lock."""
        row = self._conn.execute(
            f"SELECT * FROM {self.name} WHERE {self.pk_col} = ?", [key]
        ).fetchone()
        return dict(row) if row else None

    def _emit_op(self, op: str, row: dict | None = None, key=None):
        """Emit a single CDC op to the broadcaster if set."""
        if self._broadcaster is not None:
            self._broadcaster.publish_op(self.name, op, row=row, key=key)

    def _emit_ops(self, op: str, rows: list[dict]):
        """Emit a batched CDC op to the broadcaster if set."""
        if self._broadcaster is not None:
            self._broadcaster.publish_ops(self.name, op, rows=rows)

    def _rehash_unlocked(self):
        """Recompute hash from all rows. Caller must hold lock."""
        h = 0
        rows = self._conn.execute(f"SELECT * FROM {self.name}").fetchall()
        for row in rows:
            h ^= self._hash_row(dict(row))
        self._hash = h
