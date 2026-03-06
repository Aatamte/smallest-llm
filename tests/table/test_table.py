"""Tests for the base Table class."""

import os
import sqlite3
import tempfile
import threading

import pytest

from src.storage.table import Column, Index, Table


class SampleTable(Table):
    name = "items"
    columns = [
        Column("id", "INTEGER", primary_key=True, autoincrement=True),
        Column("name", "TEXT", not_null=True),
        Column("value", "REAL", default="0.0"),
        Column("created_at", "TEXT", default="datetime('now')"),
    ]
    indexes = [
        Index("idx_items_name", ["name"]),
    ]


class RefTable(Table):
    """Table with a foreign key reference."""
    name = "children"
    columns = [
        Column("id", "INTEGER", primary_key=True, autoincrement=True),
        Column("parent_id", "INTEGER", not_null=True, references="items(id)"),
        Column("label", "TEXT"),
    ]
    indexes = []


@pytest.fixture
def conn():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    c = sqlite3.connect(path, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    yield c
    c.close()
    os.unlink(path)


@pytest.fixture
def lock():
    return threading.Lock()


@pytest.fixture
def table(conn, lock):
    t = SampleTable(conn, lock)
    t.create()
    return t


class TestCreate:
    def test_table_exists(self, conn, table):
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='items'"
        ).fetchall()
        assert len(rows) == 1

    def test_index_exists(self, conn, table):
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_items_name'"
        ).fetchall()
        assert len(rows) == 1

    def test_create_is_idempotent(self, table):
        table.create()
        table.create()

    def test_columns_match(self, conn, table):
        info = conn.execute("PRAGMA table_info(items)").fetchall()
        col_names = [r["name"] for r in info]
        assert col_names == ["id", "name", "value", "created_at"]

    def test_foreign_key_column(self, conn, lock):
        parent = SampleTable(conn, lock)
        parent.create()
        child = RefTable(conn, lock)
        child.create()
        info = conn.execute("PRAGMA table_info(children)").fetchall()
        col_names = [r["name"] for r in info]
        assert "parent_id" in col_names


class TestInsert:
    def test_insert_returns_id(self, table):
        row_id = table.insert(name="alpha")
        assert row_id == 1

    def test_insert_increments_id(self, table):
        id1 = table.insert(name="a")
        id2 = table.insert(name="b")
        assert id2 == id1 + 1

    def test_insert_with_value(self, table):
        table.insert(name="x", value=3.14)
        rows = table.select(where="name = ?", params=["x"])
        assert rows[0]["value"] == 3.14

    def test_insert_default_value(self, table):
        table.insert(name="default_test")
        rows = table.select(where="name = ?", params=["default_test"])
        assert rows[0]["value"] == 0.0

    def test_insert_sets_created_at(self, table):
        table.insert(name="ts_test")
        rows = table.select(where="name = ?", params=["ts_test"])
        assert rows[0]["created_at"] is not None


class TestSelect:
    def test_select_all(self, table):
        table.insert(name="a")
        table.insert(name="b")
        rows = table.select()
        assert len(rows) == 2

    def test_select_with_where(self, table):
        table.insert(name="target", value=1.0)
        table.insert(name="other", value=2.0)
        rows = table.select(where="name = ?", params=["target"])
        assert len(rows) == 1
        assert rows[0]["name"] == "target"

    def test_select_with_order_by(self, table):
        table.insert(name="b", value=2.0)
        table.insert(name="a", value=1.0)
        rows = table.select(order_by="value ASC")
        assert rows[0]["name"] == "a"
        assert rows[1]["name"] == "b"

    def test_select_specific_columns(self, table):
        table.insert(name="col_test", value=5.0)
        rows = table.select(columns="name, value")
        assert "name" in rows[0]
        assert "value" in rows[0]
        assert "id" not in rows[0]

    def test_select_empty_table(self, table):
        rows = table.select()
        assert rows == []

    def test_select_returns_dicts(self, table):
        table.insert(name="dict_test")
        rows = table.select()
        assert isinstance(rows[0], dict)


class TestDelete:
    def test_delete_by_condition(self, table):
        table.insert(name="keep")
        table.insert(name="remove")
        table.delete(where="name = ?", params=["remove"])
        rows = table.select()
        assert len(rows) == 1
        assert rows[0]["name"] == "keep"

    def test_delete_all_matching(self, table):
        table.insert(name="dup")
        table.insert(name="dup")
        table.insert(name="other")
        table.delete(where="name = ?", params=["dup"])
        rows = table.select()
        assert len(rows) == 1

    def test_delete_nonexistent_is_noop(self, table):
        table.insert(name="safe")
        table.delete(where="name = ?", params=["ghost"])
        rows = table.select()
        assert len(rows) == 1


class TestUpdate:
    def test_update_single_row(self, table):
        table.insert(name="old", value=1.0)
        table.update("name = ?", "value = ?", ["new", 1.0])
        rows = table.select()
        assert rows[0]["name"] == "new"

    def test_update_multiple_fields(self, table):
        table.insert(name="orig", value=0.0)
        table.update("name = ?, value = ?", "name = ?", ["changed", 9.9, "orig"])
        rows = table.select()
        assert rows[0]["name"] == "changed"
        assert rows[0]["value"] == 9.9

    def test_update_no_match_is_noop(self, table):
        table.insert(name="untouched", value=1.0)
        table.update("value = ?", "name = ?", [99.0, "ghost"])
        rows = table.select()
        assert rows[0]["value"] == 1.0


class NoPKTable(Table):
    name = "no_pk"
    columns = [
        Column("a", "TEXT"),
        Column("b", "INTEGER"),
    ]
    indexes = []


class TwoPKTable(Table):
    name = "two_pk"
    columns = [
        Column("a", "INTEGER", primary_key=True),
        Column("b", "INTEGER", primary_key=True),
    ]
    indexes = []


class ExplicitPKTable(Table):
    """Table with explicit integer PK (no autoincrement) for upsert tests."""
    name = "explicit_pk"
    columns = [
        Column("id", "INTEGER", primary_key=True),
        Column("name", "TEXT", not_null=True),
        Column("value", "REAL", default="0.0"),
    ]
    indexes = []


class TestPKValidation:
    def test_no_pk_raises(self, conn, lock):
        with pytest.raises(ValueError, match="must have exactly one primary key column, got 0"):
            NoPKTable(conn, lock)

    def test_two_pk_raises(self, conn, lock):
        with pytest.raises(ValueError, match="must have exactly one primary key column, got 2"):
            TwoPKTable(conn, lock)

    def test_pk_col_is_set(self, table):
        assert table.pk_col == "id"


@pytest.fixture
def pk_table(conn, lock):
    t = ExplicitPKTable(conn, lock)
    t.create()
    return t


class TestUpsert:
    def test_upsert_inserts_new_row(self, pk_table):
        pk_table.upsert(id=1, name="alice", value=1.0)
        rows = pk_table.select()
        assert len(rows) == 1
        assert rows[0]["name"] == "alice"

    def test_upsert_replaces_existing(self, pk_table):
        pk_table.upsert(id=1, name="alice", value=1.0)
        pk_table.upsert(id=1, name="alice-updated", value=99.0)
        rows = pk_table.select()
        assert len(rows) == 1
        assert rows[0]["name"] == "alice-updated"
        assert rows[0]["value"] == 99.0

    def test_upsert_without_pk_raises(self, pk_table):
        with pytest.raises(ValueError, match='must include primary key column "id"'):
            pk_table.upsert(name="alice")

    def test_upsert_with_none_pk_raises(self, pk_table):
        with pytest.raises(ValueError, match='must include primary key column "id"'):
            pk_table.upsert(id=None, name="alice")


class TestUpsertMany:
    def test_batch_upserts(self, pk_table):
        pk_table.upsert_many([
            {"id": 1, "name": "a", "value": 1.0},
            {"id": 2, "name": "b", "value": 2.0},
            {"id": 3, "name": "c", "value": 3.0},
        ])
        rows = pk_table.select()
        assert len(rows) == 3

    def test_empty_list_is_noop(self, pk_table):
        pk_table.upsert_many([])
        assert pk_table.select() == []

    def test_replaces_existing(self, pk_table):
        pk_table.upsert_many([
            {"id": 1, "name": "a"},
            {"id": 2, "name": "b"},
        ])
        pk_table.upsert_many([
            {"id": 1, "name": "a-updated"},
            {"id": 2, "name": "b-updated"},
        ])
        rows = pk_table.select(order_by="id")
        assert len(rows) == 2
        assert rows[0]["name"] == "a-updated"
        assert rows[1]["name"] == "b-updated"

    def test_missing_pk_raises(self, pk_table):
        with pytest.raises(ValueError, match='must include primary key column "id"'):
            pk_table.upsert_many([{"id": 1, "name": "a"}, {"name": "b"}])


class TestClear:
    def test_removes_all_rows(self, pk_table):
        pk_table.upsert(id=1, name="a")
        pk_table.upsert(id=2, name="b")
        pk_table.clear()
        assert pk_table.select() == []

    def test_table_works_after_clear(self, pk_table):
        pk_table.upsert(id=1, name="a")
        pk_table.clear()
        pk_table.upsert(id=2, name="b")
        rows = pk_table.select()
        assert len(rows) == 1
        assert rows[0]["name"] == "b"


class TestGetAllRows:
    def test_returns_all_rows(self, pk_table):
        pk_table.upsert(id=1, name="a")
        pk_table.upsert(id=2, name="b")
        rows = pk_table.get_all_rows()
        assert len(rows) == 2

    def test_empty_table(self, pk_table):
        assert pk_table.get_all_rows() == []


class TestHashing:
    def test_hash_starts_at_zero(self, pk_table):
        assert pk_table.get_hash() == 0

    def test_hash_changes_after_upsert(self, pk_table):
        pk_table.upsert(id=1, name="a")
        assert pk_table.get_hash() != 0

    def test_hash_returns_to_zero_after_upsert_then_delete(self, pk_table):
        pk_table.upsert(id=1, name="a")
        assert pk_table.get_hash() != 0
        pk_table.delete(where="id = ?", params=[1])
        assert pk_table.get_hash() == 0

    def test_hash_is_zero_after_clear(self, pk_table):
        pk_table.upsert(id=1, name="a")
        pk_table.upsert(id=2, name="b")
        assert pk_table.get_hash() != 0
        pk_table.clear()
        assert pk_table.get_hash() == 0

    def test_hash_order_independent(self, conn, lock):
        t1 = ExplicitPKTable(conn, lock)
        t1.create()
        t1.upsert(id=1, name="a", value=1.0)
        t1.upsert(id=2, name="b", value=2.0)

        # Need a separate table (different name) in same DB
        class T2(Table):
            name = "explicit_pk2"
            columns = ExplicitPKTable.columns
            indexes = []

        t2 = T2(conn, lock)
        t2.create()
        t2.upsert(id=2, name="b", value=2.0)
        t2.upsert(id=1, name="a", value=1.0)

        assert t1.get_hash() == t2.get_hash()

    def test_upsert_same_row_twice_idempotent(self, conn, lock):
        t1 = ExplicitPKTable(conn, lock)
        t1.create()
        t1.upsert(id=1, name="a", value=5.0)
        hash_once = t1.get_hash()

        class T2(Table):
            name = "explicit_pk3"
            columns = ExplicitPKTable.columns
            indexes = []

        t2 = T2(conn, lock)
        t2.create()
        t2.upsert(id=1, name="a", value=5.0)
        t2.upsert(id=1, name="a", value=5.0)

        assert hash_once == t2.get_hash()

    def test_different_data_different_hash(self, conn, lock):
        t1 = ExplicitPKTable(conn, lock)
        t1.create()
        t1.upsert(id=1, name="a")

        class T2(Table):
            name = "explicit_pk4"
            columns = ExplicitPKTable.columns
            indexes = []

        t2 = T2(conn, lock)
        t2.create()
        t2.upsert(id=1, name="b")

        assert t1.get_hash() != t2.get_hash()

    def test_upsert_update_changes_hash(self, pk_table):
        pk_table.upsert(id=1, name="a", value=1.0)
        h1 = pk_table.get_hash()
        pk_table.upsert(id=1, name="a", value=99.0)
        h2 = pk_table.get_hash()
        assert h1 != h2

    def test_insert_also_updates_hash(self, table):
        """insert() (autoincrement) should also track hash."""
        assert table.get_hash() == 0
        table.insert(name="a")
        assert table.get_hash() != 0

    def test_delete_with_where_updates_hash(self, table):
        table.insert(name="a")
        table.insert(name="b")
        h1 = table.get_hash()
        table.delete(where="name = ?", params=["a"])
        h2 = table.get_hash()
        assert h1 != h2


class TestSchemaDict:
    def test_basic_schema(self, pk_table):
        schema = pk_table.to_schema_dict()
        assert schema["name"] == "explicit_pk"
        assert len(schema["columns"]) == 3
        pk_col = schema["columns"][0]
        assert pk_col["name"] == "id"
        assert pk_col["type"] == "INTEGER"
        assert pk_col["primaryKey"] is True

    def test_schema_with_indexes(self, table):
        schema = table.to_schema_dict()
        assert schema["name"] == "items"
        assert "indexes" in schema
        assert schema["indexes"][0]["name"] == "idx_items_name"
        assert schema["indexes"][0]["columns"] == ["name"]

    def test_schema_column_options(self, table):
        schema = table.to_schema_dict()
        name_col = next(c for c in schema["columns"] if c["name"] == "name")
        assert name_col["notNull"] is True
        value_col = next(c for c in schema["columns"] if c["name"] == "value")
        assert value_col["default"] == "0.0"
