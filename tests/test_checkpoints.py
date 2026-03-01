"""Tests for the checkpoint system — DB storage, API endpoints, CheckpointManager."""

import json
import os
import tempfile

import pytest

from src.storage.database import Database


# ── Fixtures ──────────────────────────────────────────────


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def db(db_path):
    database = Database(db_path)
    yield database
    database.close()


@pytest.fixture
def run_id(db):
    return db.create_run("checkpoint-test", {"training": {"max_steps": 1000}})


# ── Database: log_checkpoint + get_checkpoints ───────────


class TestDatabaseCheckpoints:
    def test_log_and_retrieve(self, db, run_id):
        db.log_checkpoint(run_id, step=100, path="/tmp/ckpt-100.pt",
                          metrics={"val/loss": 3.5}, is_best=False)
        db.log_checkpoint(run_id, step=200, path="/tmp/ckpt-200.pt",
                          metrics={"val/loss": 2.8}, is_best=True)

        checkpoints = db.get_checkpoints(run_id)
        assert len(checkpoints) == 2

        c1, c2 = checkpoints
        assert c1["step"] == 100
        assert c1["path"] == "/tmp/ckpt-100.pt"
        assert c1["is_best"] == 0
        assert json.loads(c1["metrics"])["val/loss"] == 3.5

        assert c2["step"] == 200
        assert c2["is_best"] == 1
        assert json.loads(c2["metrics"])["val/loss"] == 2.8

    def test_no_checkpoints(self, db, run_id):
        checkpoints = db.get_checkpoints(run_id)
        assert checkpoints == []

    def test_none_metrics(self, db, run_id):
        db.log_checkpoint(run_id, step=50, path="/tmp/ckpt-50.pt",
                          metrics=None, is_best=False)
        checkpoints = db.get_checkpoints(run_id)
        assert len(checkpoints) == 1
        assert checkpoints[0]["metrics"] is None

    def test_ordered_by_step(self, db, run_id):
        db.log_checkpoint(run_id, step=300, path="/tmp/ckpt-300.pt")
        db.log_checkpoint(run_id, step=100, path="/tmp/ckpt-100.pt")
        db.log_checkpoint(run_id, step=200, path="/tmp/ckpt-200.pt")

        checkpoints = db.get_checkpoints(run_id)
        steps = [c["step"] for c in checkpoints]
        assert steps == [100, 200, 300]

    def test_multiple_runs_isolated(self, db):
        r1 = db.create_run("run1", {})
        r2 = db.create_run("run2", {})
        db.log_checkpoint(r1, step=100, path="/tmp/r1-100.pt")
        db.log_checkpoint(r2, step=100, path="/tmp/r2-100.pt")
        db.log_checkpoint(r2, step=200, path="/tmp/r2-200.pt")

        assert len(db.get_checkpoints(r1)) == 1
        assert len(db.get_checkpoints(r2)) == 2

    def test_multiple_best_flags(self, db, run_id):
        """Multiple checkpoints can be marked as best (DB doesn't enforce uniqueness)."""
        db.log_checkpoint(run_id, step=100, path="/tmp/a.pt", is_best=True)
        db.log_checkpoint(run_id, step=200, path="/tmp/b.pt", is_best=True)
        db.log_checkpoint(run_id, step=300, path="/tmp/c.pt", is_best=False)

        checkpoints = db.get_checkpoints(run_id)
        best_count = sum(1 for c in checkpoints if c["is_best"])
        assert best_count == 2


# ── API: GET /api/runs/{run_id}/checkpoints ──────────────


class TestCheckpointAPI:
    @pytest.fixture
    def client(self, db):
        from src.storage.eval_db import EvalDatabase
        import src.server.app as server_app

        eval_fd, eval_path = tempfile.mkstemp(suffix=".eval.db")
        os.close(eval_fd)
        eval_db = EvalDatabase(eval_path)

        old_db = server_app.run_manager.db
        old_eval_db = server_app.eval_db
        server_app.run_manager.db = db
        server_app.eval_db = eval_db

        from fastapi.testclient import TestClient
        client = TestClient(server_app.app)
        yield client

        server_app.run_manager.db = old_db
        server_app.eval_db = old_eval_db
        eval_db.close()
        os.unlink(eval_path)

    def test_returns_checkpoints_with_parsed_metrics(self, client, db, run_id):
        db.log_checkpoint(run_id, step=100, path="/tmp/ckpt-100.pt",
                          metrics={"val/loss": 3.5}, is_best=False)
        db.log_checkpoint(run_id, step=200, path="/tmp/ckpt-200.pt",
                          metrics={"val/loss": 2.8}, is_best=True)

        resp = client.get(f"/api/runs/{run_id}/checkpoints")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

        # Metrics should be parsed dicts, not JSON strings
        assert data[0]["metrics"]["val/loss"] == 3.5
        assert data[1]["metrics"]["val/loss"] == 2.8
        assert data[1]["is_best"] == 1

    def test_empty_checkpoints(self, client, db, run_id):
        resp = client.get(f"/api/runs/{run_id}/checkpoints")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_null_metrics_returns_empty_dict(self, client, db, run_id):
        db.log_checkpoint(run_id, step=50, path="/tmp/ckpt.pt",
                          metrics=None, is_best=False)
        resp = client.get(f"/api/runs/{run_id}/checkpoints")
        data = resp.json()
        assert data[0]["metrics"] == {}

    def test_nonexistent_run_returns_empty(self, client):
        resp = client.get("/api/runs/9999/checkpoints")
        assert resp.status_code == 200
        assert resp.json() == []


# ── CheckpointManager ───────────────────────────────────


class TestCheckpointManager:
    @pytest.fixture
    def ckpt_dir(self):
        d = tempfile.mkdtemp()
        yield d
        import shutil
        shutil.rmtree(d, ignore_errors=True)

    @pytest.fixture
    def manager(self, ckpt_dir, db, run_id):
        from src.config.base import CheckpointConfig, ExperimentConfig
        from src.training.checkpointing import CheckpointManager

        config = CheckpointConfig(
            save_dir=ckpt_dir, keep_last_n=3,
            save_best=True, best_metric="val/loss", best_mode="min",
        )
        experiment_config = ExperimentConfig(name="test-ckpt")
        return CheckpointManager(config, experiment_config, db=db, run_id=run_id)

    @pytest.fixture
    def dummy_model(self):
        import torch
        return torch.nn.Linear(4, 4)

    @pytest.fixture
    def dummy_optimizer(self, dummy_model):
        import torch
        return torch.optim.Adam(dummy_model.parameters())

    @pytest.fixture
    def dummy_scheduler(self, dummy_optimizer):
        import torch
        return torch.optim.lr_scheduler.StepLR(dummy_optimizer, step_size=10)

    def test_save_creates_file(self, manager, dummy_model, dummy_optimizer, dummy_scheduler):
        manager.save(100, dummy_model, dummy_optimizer, dummy_scheduler,
                     {"val/loss": 3.0}, tokens_seen=10000)

        expected_path = os.path.join(manager.save_dir, "checkpoint-100.pt")
        assert os.path.exists(expected_path)

    def test_save_logs_to_db(self, manager, dummy_model, dummy_optimizer, dummy_scheduler, db, run_id):
        manager.save(100, dummy_model, dummy_optimizer, dummy_scheduler,
                     {"val/loss": 3.0}, tokens_seen=10000)

        checkpoints = db.get_checkpoints(run_id)
        assert len(checkpoints) == 1
        assert checkpoints[0]["step"] == 100

    def test_best_checkpoint_saved(self, manager, dummy_model, dummy_optimizer, dummy_scheduler, db, run_id):
        manager.save(100, dummy_model, dummy_optimizer, dummy_scheduler,
                     {"val/loss": 3.0}, tokens_seen=10000)
        manager.save(200, dummy_model, dummy_optimizer, dummy_scheduler,
                     {"val/loss": 2.5}, tokens_seen=20000)

        best_path = os.path.join(manager.save_dir, "best.pt")
        assert os.path.exists(best_path)

        checkpoints = db.get_checkpoints(run_id)
        # First was best when saved, second is also best (lower loss)
        assert checkpoints[0]["is_best"] == 1  # 3.0 was best at the time
        assert checkpoints[1]["is_best"] == 1  # 2.5 is better

    def test_non_improving_not_best(self, manager, dummy_model, dummy_optimizer, dummy_scheduler, db, run_id):
        manager.save(100, dummy_model, dummy_optimizer, dummy_scheduler,
                     {"val/loss": 2.0}, tokens_seen=10000)
        manager.save(200, dummy_model, dummy_optimizer, dummy_scheduler,
                     {"val/loss": 3.0}, tokens_seen=20000)

        checkpoints = db.get_checkpoints(run_id)
        assert checkpoints[0]["is_best"] == 1  # 2.0 is best
        assert checkpoints[1]["is_best"] == 0  # 3.0 is worse

    def test_rotation_keeps_last_n(self, manager, dummy_model, dummy_optimizer, dummy_scheduler):
        for step in [100, 200, 300, 400, 500]:
            manager.save(step, dummy_model, dummy_optimizer, dummy_scheduler,
                         {"val/loss": 5.0 - step * 0.01}, tokens_seen=step * 100)

        # keep_last_n=3, so only 3 should remain
        remaining = manager._list_checkpoints()
        assert len(remaining) == 3
        # Should keep the latest 3
        assert "checkpoint-300" in remaining[0]
        assert "checkpoint-400" in remaining[1]
        assert "checkpoint-500" in remaining[2]

    def test_find_latest(self, manager, dummy_model, dummy_optimizer, dummy_scheduler):
        assert manager.find_latest() is None

        manager.save(100, dummy_model, dummy_optimizer, dummy_scheduler,
                     {"val/loss": 3.0}, tokens_seen=10000)
        manager.save(200, dummy_model, dummy_optimizer, dummy_scheduler,
                     {"val/loss": 2.5}, tokens_seen=20000)

        latest = manager.find_latest()
        assert latest is not None
        assert "checkpoint-200" in latest

    def test_no_db_still_saves_file(self, ckpt_dir, dummy_model, dummy_optimizer, dummy_scheduler):
        """CheckpointManager works without a DB (db=None)."""
        from src.config.base import CheckpointConfig, ExperimentConfig
        from src.training.checkpointing import CheckpointManager

        config = CheckpointConfig(save_dir=ckpt_dir, keep_last_n=3)
        manager = CheckpointManager(config, ExperimentConfig(name="no-db"))

        manager.save(100, dummy_model, dummy_optimizer, dummy_scheduler,
                     {"val/loss": 3.0}, tokens_seen=10000)

        assert os.path.exists(os.path.join(manager.save_dir, "checkpoint-100.pt"))
