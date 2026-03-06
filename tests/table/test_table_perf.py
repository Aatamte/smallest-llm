"""Performance tests for Table operations and CDC batching."""

import os
import sqlite3
import tempfile
import threading
import time

import pytest

from src.storage.table import Column, Index, Table


class MetricsLike(Table):
    """Mimics the metrics table schema."""
    name = "metrics_perf"
    columns = [
        Column("id", "INTEGER", primary_key=True, autoincrement=True),
        Column("run_id", "INTEGER", not_null=True),
        Column("step", "INTEGER", not_null=True),
        Column("key", "TEXT", not_null=True),
        Column("value", "REAL", not_null=True),
        Column("timestamp", "TEXT", default="datetime('now')"),
    ]
    indexes = [
        Index("idx_mp_run_step", ["run_id", "step"]),
    ]


class FakeBroadcaster:
    """Counts messages and ops."""
    def __init__(self):
        self.message_count = 0
        self.total_rows = 0

    def publish_op(self, table, op, row=None, key=None):
        self.message_count += 1
        self.total_rows += 1

    def publish_ops(self, table, op, rows=None):
        self.message_count += 1
        self.total_rows += len(rows or [])


@pytest.fixture
def setup():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    lock = threading.Lock()
    table = MetricsLike(conn, lock)
    table.create()
    bc = FakeBroadcaster()
    table.set_broadcaster(bc)
    yield table, bc
    conn.close()
    os.unlink(path)


# ── CDC message count tests ──────────────────────────────


class TestCDCBatching:
    def test_insert_many_emits_one_message(self, setup):
        table, bc = setup
        rows = [{"run_id": 1, "step": 0, "key": f"metric_{i}", "value": float(i)} for i in range(10)]
        table.insert_many(rows)
        assert bc.message_count == 1
        assert bc.total_rows == 10

    def test_individual_inserts_emit_n_messages(self, setup):
        table, bc = setup
        for i in range(10):
            table.insert(run_id=1, step=0, key=f"metric_{i}", value=float(i))
        assert bc.message_count == 10
        assert bc.total_rows == 10

    def test_upsert_many_emits_one_message(self, setup):
        table, bc = setup
        # First insert to get IDs
        ids = table.insert_many([
            {"run_id": 1, "step": 0, "key": f"m{i}", "value": float(i)} for i in range(5)
        ])
        bc.message_count = 0
        bc.total_rows = 0
        # Now upsert_many with those IDs
        table.upsert_many([
            {"id": ids[i], "run_id": 1, "step": 0, "key": f"m{i}", "value": float(i * 10)} for i in range(5)
        ])
        assert bc.message_count == 1
        assert bc.total_rows == 5

    def test_update_emits_one_message(self, setup):
        table, bc = setup
        table.insert_many([
            {"run_id": 1, "step": 0, "key": f"m{i}", "value": float(i)} for i in range(5)
        ])
        bc.message_count = 0
        bc.total_rows = 0
        table.update("value = ?", "run_id = ?", [99.0, 1])
        assert bc.message_count == 1
        assert bc.total_rows == 5


# ── Performance benchmarks ───────────────────────────────


class TestInsertPerformance:
    """Benchmarks: insert_many vs individual insert."""

    N = 500  # rows per batch (simulates ~50 steps × 10 metrics)

    def test_insert_many_faster_than_individual(self, setup):
        table, bc = setup
        rows = [{"run_id": 1, "step": i // 10, "key": f"m{i % 10}", "value": float(i)} for i in range(self.N)]

        # Benchmark insert_many
        t0 = time.perf_counter()
        table.insert_many(rows)
        t_batch = time.perf_counter() - t0

        # Reset
        table.clear()
        bc.message_count = 0

        # Benchmark individual inserts
        t0 = time.perf_counter()
        for row in rows:
            table.insert(**row)
        t_individual = time.perf_counter() - t0

        speedup = t_individual / t_batch
        print(f"\n  insert_many: {t_batch:.4f}s | individual: {t_individual:.4f}s | speedup: {speedup:.1f}x")
        # insert_many should be meaningfully faster (at least 1.5x)
        assert speedup > 1.5, f"Expected insert_many to be >=1.5x faster, got {speedup:.2f}x"

    def test_insert_many_cdc_messages(self, setup):
        """insert_many for N rows should emit 1 CDC message, not N."""
        table, bc = setup
        rows = [{"run_id": 1, "step": 0, "key": f"m{i}", "value": float(i)} for i in range(self.N)]
        table.insert_many(rows)
        assert bc.message_count == 1, f"Expected 1 CDC message, got {bc.message_count}"

    def test_hash_consistent_between_insert_and_insert_many(self):
        """insert and insert_many should produce the same hash for the same data."""
        fd1, path1 = tempfile.mkstemp(suffix=".db")
        fd2, path2 = tempfile.mkstemp(suffix=".db")
        os.close(fd1)
        os.close(fd2)

        lock = threading.Lock()
        conn1 = sqlite3.connect(path1, check_same_thread=False)
        conn1.row_factory = sqlite3.Row
        conn2 = sqlite3.connect(path2, check_same_thread=False)
        conn2.row_factory = sqlite3.Row

        t1 = MetricsLike(conn1, lock)
        t1.create()

        # Use a different table name for t2 since same schema
        class MetricsLike2(Table):
            name = "metrics_perf2"
            columns = MetricsLike.columns
            indexes = []

        lock2 = threading.Lock()
        t2 = MetricsLike2(conn2, lock2)
        t2.create()

        rows = [{"run_id": 1, "step": 0, "key": f"m{i}", "value": float(i)} for i in range(20)]

        # Individual inserts
        for row in rows:
            t1.insert(**row)

        # Batch insert
        t2.insert_many(rows)

        assert t1.get_hash() == t2.get_hash(), "Hashes should match between insert and insert_many"

        conn1.close()
        conn2.close()
        os.unlink(path1)
        os.unlink(path2)


class TestInitHashPerformance:
    """Benchmark _init_hash on a table with existing data."""

    def test_init_hash_reasonable_time(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        lock = threading.Lock()

        table = MetricsLike(conn, lock)
        table.create()

        # Insert 10K rows (simulates moderate usage)
        N = 10000
        rows = [{"run_id": 1, "step": i // 10, "key": f"m{i % 10}", "value": float(i)} for i in range(N)]
        table.insert_many(rows)
        expected_hash = table.get_hash()

        # Now reset hash and time _init_hash
        table._hash = 0
        t0 = time.perf_counter()
        table._init_hash()
        t_hash = time.perf_counter() - t0

        print(f"\n  _init_hash({N} rows): {t_hash:.4f}s ({N / t_hash:.0f} rows/s)")
        assert table.get_hash() == expected_hash, "Hash after _init_hash should match"
        # Should complete in under 5 seconds for 10K rows
        assert t_hash < 5.0, f"_init_hash took {t_hash:.2f}s for {N} rows — too slow"

        conn.close()
        os.unlink(path)
