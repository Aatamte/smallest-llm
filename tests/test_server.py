"""Tests for the dashboard backend API."""

import json
import os
import tempfile
import threading
import time

import pytest
from fastapi.testclient import TestClient

from src.server.broadcast import Broadcaster
from src.storage.database import Database
from src.storage.eval_db import EvalDatabase

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
def eval_db_path():
    fd, path = tempfile.mkstemp(suffix=".eval.db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def eval_db(eval_db_path):
    database = EvalDatabase(eval_db_path)
    yield database
    database.close()


@pytest.fixture
def seeded_db(db, eval_db):
    """DB with a run, metrics, evals, and checkpoints."""
    config = {
        "training": {"max_steps": 1000},
        "optimizer": {"lr": 3e-4},
    }
    env = {"torch": "2.8.0", "python": "3.9.13"}
    run_id = db.create_run("test-experiment", config, env)

    # Log training metrics at several steps
    db.log_metrics(run_id, step=0, metrics={
        "train/loss": 4.2, "train/lr": 1e-5, "train/grad_norm": 1.5,
        "train/tokens_seen": 16384, "train/step_time": 0.05,
    })
    db.log_metrics(run_id, step=10, metrics={
        "train/loss": 3.8, "train/lr": 3e-4, "train/grad_norm": 0.8,
        "train/tokens_seen": 180224, "train/step_time": 0.04,
    })
    # Val loss at step 10
    db.log_metrics(run_id, step=10, metrics={"val/loss": 4.0})
    db.log_metrics(run_id, step=20, metrics={
        "train/loss": 3.2, "train/lr": 2.9e-4, "train/grad_norm": 0.5,
        "train/tokens_seen": 344064, "train/step_time": 0.04,
    })

    # Eval results (now in eval_db)
    eval_db.log_eval(
        task="perplexity", metrics={"perplexity": 45.2, "bpc": 1.3},
        metadata={"num_tokens": 5000}, run_id=run_id, step=20,
    )
    eval_db.log_eval(
        task="blimp", metrics={"accuracy": 0.65},
        metadata={"num_pairs": 3350}, run_id=run_id, step=20,
    )

    # Checkpoint
    db.log_checkpoint(run_id, step=20, path="/tmp/ckpt-20.pt",
                      metrics={"val/loss": 4.0}, is_best=True)

    return db, run_id


@pytest.fixture
def client(db_path, seeded_db, eval_db):
    """TestClient wired to a seeded database."""
    import src.server.app as server_app
    old_db = server_app.run_manager.db
    old_eval_db = server_app.eval_db
    # Replace with test DBs
    server_app.run_manager.db = seeded_db[0]
    server_app.eval_db = eval_db
    yield TestClient(server_app.app)
    server_app.run_manager.db = old_db
    server_app.eval_db = old_eval_db


# ── GET /api/runs ─────────────────────────────────────────


class TestListRuns:
    def test_returns_runs(self, client):
        resp = client.get("/api/runs")
        assert resp.status_code == 200
        runs = resp.json()
        assert len(runs) == 1
        assert runs[0]["name"] == "test-experiment"
        assert runs[0]["status"] == "running"

    def test_empty_db(self):
        import src.server.app as server_app
        import tempfile, os
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        empty_db = Database(path)
        old_db = server_app.run_manager.db
        server_app.run_manager.db = empty_db
        c = TestClient(server_app.app)
        resp = c.get("/api/runs")
        assert resp.status_code == 200
        assert resp.json() == []
        server_app.run_manager.db = old_db
        empty_db.close()
        os.unlink(path)


# ── GET /api/runs/{run_id} ────────────────────────────────


class TestGetRun:
    def test_returns_run_with_parsed_config(self, client, seeded_db):
        _, run_id = seeded_db
        resp = client.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "test-experiment"
        assert data["config"]["optimizer"]["lr"] == 3e-4
        assert data["env"]["torch"] == "2.8.0"

    def test_run_not_found(self, client):
        resp = client.get("/api/runs/999")
        assert resp.status_code == 404


# ── GET /api/runs/{run_id}/metrics ────────────────────────


class TestGetMetrics:
    def test_returns_all_metrics(self, client, seeded_db):
        _, run_id = seeded_db
        resp = client.get(f"/api/runs/{run_id}/metrics")
        assert resp.status_code == 200
        metrics = resp.json()
        assert len(metrics) > 0
        steps = set(m["step"] for m in metrics)
        assert steps == {0, 10, 20}

    def test_filter_by_key(self, client, seeded_db):
        _, run_id = seeded_db
        resp = client.get(f"/api/runs/{run_id}/metrics?key=train/loss")
        assert resp.status_code == 200
        metrics = resp.json()
        assert all(m["key"] == "train/loss" for m in metrics)
        assert len(metrics) == 3

    def test_filter_val_loss(self, client, seeded_db):
        _, run_id = seeded_db
        resp = client.get(f"/api/runs/{run_id}/metrics?key=val/loss")
        metrics = resp.json()
        assert len(metrics) == 1
        assert metrics[0]["step"] == 10
        assert metrics[0]["value"] == 4.0


# ── GET /api/runs/{run_id}/evals ──────────────────────────


class TestGetEvals:
    def test_returns_evals_with_parsed_json(self, client, seeded_db):
        _, run_id = seeded_db
        resp = client.get(f"/api/runs/{run_id}/evals")
        assert resp.status_code == 200
        evals = resp.json()
        assert len(evals) == 2

        ppl = next(e for e in evals if e["task"] == "perplexity")
        assert ppl["metrics"]["perplexity"] == 45.2
        assert ppl["metadata"]["num_tokens"] == 5000

        blimp = next(e for e in evals if e["task"] == "blimp")
        assert blimp["metrics"]["accuracy"] == 0.65

    def test_no_evals(self, client, seeded_db):
        db = seeded_db[0]
        run_id = db.create_run("empty-run", {})
        resp = client.get(f"/api/runs/{run_id}/evals")
        assert resp.json() == []


# ── GET /api/runs/{run_id}/checkpoints ────────────────────


class TestGetCheckpoints:
    def test_returns_checkpoints(self, client, seeded_db):
        _, run_id = seeded_db
        resp = client.get(f"/api/runs/{run_id}/checkpoints")
        assert resp.status_code == 200
        checkpoints = resp.json()
        assert len(checkpoints) == 1
        assert checkpoints[0]["step"] == 20
        assert checkpoints[0]["is_best"] == 1
        assert checkpoints[0]["metrics"]["val/loss"] == 4.0


# ── GET /api/runs/{run_id}/state ──────────────────────────


class TestGetTrainingState:
    def test_returns_full_state(self, client, seeded_db):
        _, run_id = seeded_db
        resp = client.get(f"/api/runs/{run_id}/state")
        assert resp.status_code == 200
        state = resp.json()

        assert state["experimentName"] == "test-experiment"
        assert state["status"] == "training"  # mapped from "running"
        assert state["maxSteps"] == 1000

        steps = state["steps"]
        assert len(steps) == 3

        s0 = steps[0]
        assert s0["step"] == 0
        assert s0["trainLoss"] == 4.2
        assert s0["lr"] == 1e-5
        assert s0["gradNorm"] == 1.5
        assert "bpc" in s0

        s10 = steps[1]
        assert s10["step"] == 10
        assert "valLoss" in s10
        assert s10["valLoss"] == 4.0

        assert state["currentStep"] == 20
        assert state["currentTrainLoss"] == 3.2
        assert state["bestValLoss"] == 4.0

    def test_state_empty_run(self, client, seeded_db):
        db = seeded_db[0]
        run_id = db.create_run("empty", {"training": {"max_steps": 500}})
        resp = client.get(f"/api/runs/{run_id}/state")
        state = resp.json()
        assert state["experimentName"] == "empty"
        assert state["steps"] == []
        assert state["currentStep"] == 0


# ── GET /api/active-run ──────────────────────────────────


class TestActiveRun:
    def test_no_active_run(self, client):
        resp = client.get("/api/active-run")
        assert resp.status_code == 200
        assert resp.json() is None

    def test_active_run_after_manual_set(self, client):
        import src.server.app as server_app
        mgr = server_app.run_manager
        # Simulate an active run
        mgr._active_run_id = 42
        mgr._active_thread = threading.Thread(target=lambda: None)
        mgr._active_thread.start()
        mgr._active_thread.join()  # let it finish but keep ref

        resp = client.get("/api/active-run")
        data = resp.json()
        assert data["run_id"] == 42

        # Cleanup
        mgr._active_run_id = None
        mgr._active_trainer = None
        mgr._active_thread = None


# ── POST /api/runs/stop ──────────────────────────────────


class TestStopRun:
    def test_stop_no_active_run(self, client):
        resp = client.post("/api/runs/stop")
        assert resp.status_code == 404


# ── WebSocket /ws ─────────────────────────────────────────


class TestWebSocket:
    def test_receives_broadcast_message(self, client):
        from src.server.broadcast import broadcaster

        with client.websocket_connect("/ws") as ws:
            broadcaster.publish({"type": "step", "data": {"step": 1, "trainLoss": 3.5}})
            msg = ws.receive_json()
            assert msg["type"] == "step"
            assert msg["data"]["step"] == 1
            assert msg["data"]["trainLoss"] == 3.5

    def test_receives_multiple_message_types(self, client):
        from src.server.broadcast import broadcaster

        with client.websocket_connect("/ws") as ws:
            broadcaster.publish({"type": "step", "data": {"step": 1}})
            broadcaster.publish({"type": "layers", "data": [{"name": "embed", "gradNorm": 0.1}]})
            broadcaster.publish({"type": "status", "data": "complete"})

            m1 = ws.receive_json()
            assert m1["type"] == "step"

            m2 = ws.receive_json()
            assert m2["type"] == "layers"
            assert m2["data"][0]["name"] == "embed"

            m3 = ws.receive_json()
            assert m3["type"] == "status"
            assert m3["data"] == "complete"


# ── Broadcaster unit tests ────────────────────────────────


class TestBroadcaster:
    def test_publish_subscribe(self):
        import asyncio

        b = Broadcaster()
        received = []

        async def run():
            b.set_loop(asyncio.get_running_loop())
            sub = b.subscribe()

            async def publish_later():
                await asyncio.sleep(0.01)
                b.publish({"type": "step", "data": {"step": 1}})
                b.publish({"type": "step", "data": {"step": 2}})

            asyncio.create_task(publish_later())

            async for msg in sub:
                received.append(json.loads(msg))
                if len(received) >= 2:
                    break

        asyncio.run(run())
        assert len(received) == 2
        assert received[0]["data"]["step"] == 1
        assert received[1]["data"]["step"] == 2

    def test_no_subscribers_no_error(self):
        b = Broadcaster()
        b.publish({"type": "step", "data": {}})
