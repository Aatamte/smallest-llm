"""Tests for MainDatabase (smallest_llm.db)."""

import os
import tempfile

import pytest

from src.storage import MainDatabase


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = MainDatabase(path)
    yield database
    database.close()
    os.unlink(path)


@pytest.fixture
def run_id(db):
    return db.create_run("test-run", {"lr": 0.001}, {"torch": "2.0"})


# ── Runs ─────────────────────────────────────────────────


class TestRuns:
    def test_create_run(self, db):
        rid = db.create_run("exp-1", {"lr": 0.01})
        assert rid == 1

    def test_create_run_increments(self, db):
        r1 = db.create_run("a", {})
        r2 = db.create_run("b", {})
        assert r2 == r1 + 1

    def test_get_run(self, db, run_id):
        run = db.get_run(run_id)
        assert run is not None
        assert run["name"] == "test-run"
        assert run["status"] == "running"

    def test_get_run_not_found(self, db):
        assert db.get_run(9999) is None

    def test_list_runs(self, db, run_id):
        runs = db.list_runs()
        assert len(runs) == 1
        assert runs[0]["id"] == run_id

    def test_list_runs_empty(self, db):
        assert db.list_runs() == []

    def test_rename_run(self, db, run_id):
        db.rename_run(run_id, "new-name")
        run = db.get_run(run_id)
        assert run["name"] == "new-name"

    def test_finish_run(self, db, run_id):
        db.finish_run(run_id, "completed")
        run = db.get_run(run_id)
        assert run["status"] == "completed"
        assert run["finished_at"] is not None

    def test_mark_stale_runs(self, db):
        r1 = db.create_run("stale-1", {})
        r2 = db.create_run("stale-2", {})
        db.finish_run(r2, "completed")
        stale = db.mark_stale_runs()
        assert r1 in stale
        assert r2 not in stale
        run = db.get_run(r1)
        assert run["status"] == "failed"

    def test_delete_run(self, db, run_id):
        db.log_metrics(run_id, 1, {"loss": 2.0})
        db.log_checkpoint(run_id, 1, "/tmp/ckpt.pt")
        db.delete_run(run_id)
        assert db.get_run(run_id) is None
        assert db.get_metrics(run_id) == []
        assert db.get_checkpoints(run_id) == []


# ── Metrics ──────────────────────────────────────────────


class TestMetrics:
    def test_log_and_get(self, db, run_id):
        db.log_metrics(run_id, 10, {"train/loss": 2.5, "train/lr": 0.001})
        rows = db.get_metrics(run_id)
        assert len(rows) == 2
        keys = {r["key"] for r in rows}
        assert keys == {"train/loss", "train/lr"}

    def test_get_by_key(self, db, run_id):
        db.log_metrics(run_id, 1, {"a": 1.0, "b": 2.0})
        rows = db.get_metrics(run_id, key="a")
        assert len(rows) == 1
        assert rows[0]["key"] == "a"

    def test_filters_nan_inf(self, db, run_id):
        db.log_metrics(run_id, 1, {
            "ok": 1.0,
            "nan": float("nan"),
            "inf": float("inf"),
            "neg_inf": float("-inf"),
        })
        rows = db.get_metrics(run_id)
        assert len(rows) == 1
        assert rows[0]["key"] == "ok"

    def test_filters_non_numeric(self, db, run_id):
        db.log_metrics(run_id, 1, {"ok": 1.0, "bad": "string"})
        rows = db.get_metrics(run_id)
        assert len(rows) == 1

    def test_empty_metrics_is_noop(self, db, run_id):
        db.log_metrics(run_id, 1, {})
        assert db.get_metrics(run_id) == []

    def test_ordered_by_step(self, db, run_id):
        db.log_metrics(run_id, 20, {"loss": 1.0})
        db.log_metrics(run_id, 10, {"loss": 2.0})
        rows = db.get_metrics(run_id)
        assert rows[0]["step"] == 10
        assert rows[1]["step"] == 20


# ── Checkpoints ──────────────────────────────────────────


class TestCheckpoints:
    def test_log_and_get(self, db, run_id):
        db.log_checkpoint(run_id, 100, "/tmp/ckpt-100.pt", {"val/loss": 3.0})
        rows = db.get_checkpoints(run_id)
        assert len(rows) == 1
        assert rows[0]["step"] == 100
        assert rows[0]["path"] == "/tmp/ckpt-100.pt"

    def test_is_best_flag(self, db, run_id):
        db.log_checkpoint(run_id, 100, "/tmp/a.pt", is_best=False)
        db.log_checkpoint(run_id, 200, "/tmp/b.pt", is_best=True)
        rows = db.get_checkpoints(run_id)
        assert rows[0]["is_best"] == 0
        assert rows[1]["is_best"] == 1

    def test_null_metrics(self, db, run_id):
        db.log_checkpoint(run_id, 50, "/tmp/ckpt.pt", metrics=None)
        rows = db.get_checkpoints(run_id)
        assert rows[0]["metrics"] is None

    def test_ordered_by_step(self, db, run_id):
        db.log_checkpoint(run_id, 200, "/tmp/b.pt")
        db.log_checkpoint(run_id, 100, "/tmp/a.pt")
        rows = db.get_checkpoints(run_id)
        assert rows[0]["step"] == 100
        assert rows[1]["step"] == 200

    def test_empty(self, db, run_id):
        assert db.get_checkpoints(run_id) == []


# ── Models ───────────────────────────────────────────────


class TestModels:
    def test_create_and_list(self, db, run_id):
        mid = db.create_model(run_id, "my-model", "/models/my-model")
        assert mid == 1
        models = db.list_models()
        assert len(models) == 1
        assert models[0]["name"] == "my-model"

    def test_get_model(self, db, run_id):
        mid = db.create_model(run_id, "m", "/m")
        model = db.get_model(mid)
        assert model is not None
        assert model["name"] == "m"

    def test_get_model_not_found(self, db):
        assert db.get_model(999) is None

    def test_get_by_name(self, db):
        db.create_model(None, "findme", "/findme")
        model = db.get_model_by_name("findme")
        assert model is not None
        assert model["path"] == "/findme"

    def test_get_by_name_not_found(self, db):
        assert db.get_model_by_name("nope") is None

    def test_delete_model(self, db):
        mid = db.create_model(None, "del", "/del")
        db.delete_model(mid)
        assert db.get_model(mid) is None

    def test_sync_dir(self, db, tmp_path):
        os.makedirs(tmp_path / "model_a")
        os.makedirs(tmp_path / "model_b")
        db.sync_models_dir(str(tmp_path))
        models = db.list_models()
        names = {m["name"] for m in models}
        assert names == {"model_a", "model_b"}

    def test_sync_dir_removes_stale(self, db, tmp_path):
        os.makedirs(tmp_path / "exists")
        db.sync_models_dir(str(tmp_path))
        assert len(db.list_models()) == 1
        # Remove directory and re-sync
        os.rmdir(tmp_path / "exists")
        db.sync_models_dir(str(tmp_path))
        assert len(db.list_models()) == 0
