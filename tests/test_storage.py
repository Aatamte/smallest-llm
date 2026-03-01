"""Tests for the SQLite storage layer."""

import json
import os
import tempfile

import pytest

from src.storage.database import Database


@pytest.fixture
def db():
    """Create a temporary database for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = Database(path)
    yield database
    database.close()
    os.unlink(path)


class TestRuns:
    def test_create_run(self, db):
        run_id = db.create_run("test-run", {"lr": 1e-3}, {"torch": "2.0"})
        assert run_id == 1

    def test_create_multiple_runs(self, db):
        id1 = db.create_run("run-1", {"lr": 1e-3})
        id2 = db.create_run("run-2", {"lr": 1e-4})
        assert id1 == 1
        assert id2 == 2

    def test_get_run(self, db):
        run_id = db.create_run("test-run", {"lr": 1e-3}, {"torch": "2.0"})
        run = db.get_run(run_id)
        assert run["name"] == "test-run"
        assert run["status"] == "running"
        assert json.loads(run["config"]) == {"lr": 1e-3}
        assert json.loads(run["env"]) == {"torch": "2.0"}

    def test_get_run_not_found(self, db):
        assert db.get_run(999) is None

    def test_finish_run(self, db):
        run_id = db.create_run("test-run", {})
        db.finish_run(run_id, status="completed")
        run = db.get_run(run_id)
        assert run["status"] == "completed"
        assert run["finished_at"] is not None

    def test_finish_run_failed(self, db):
        run_id = db.create_run("test-run", {})
        db.finish_run(run_id, status="failed")
        assert db.get_run(run_id)["status"] == "failed"

    def test_list_runs(self, db):
        db.create_run("run-a", {})
        db.create_run("run-b", {})
        runs = db.list_runs()
        assert len(runs) == 2
        assert runs[0]["name"] == "run-a"
        assert runs[1]["name"] == "run-b"

    def test_list_runs_empty(self, db):
        assert db.list_runs() == []


class TestMetrics:
    def test_log_and_get_metrics(self, db):
        run_id = db.create_run("test", {})
        db.log_metrics(run_id, step=0, metrics={"train/loss": 2.5, "train/lr": 1e-3})
        db.log_metrics(run_id, step=1, metrics={"train/loss": 2.3, "train/lr": 9e-4})

        all_metrics = db.get_metrics(run_id)
        assert len(all_metrics) == 4  # 2 keys x 2 steps

    def test_get_metrics_by_key(self, db):
        run_id = db.create_run("test", {})
        db.log_metrics(run_id, step=0, metrics={"train/loss": 2.5, "train/lr": 1e-3})
        db.log_metrics(run_id, step=1, metrics={"train/loss": 2.3, "train/lr": 9e-4})

        loss_metrics = db.get_metrics(run_id, key="train/loss")
        assert len(loss_metrics) == 2
        assert loss_metrics[0]["value"] == 2.5
        assert loss_metrics[1]["value"] == 2.3

    def test_log_metrics_skips_non_numeric(self, db):
        run_id = db.create_run("test", {})
        db.log_metrics(run_id, step=0, metrics={"train/loss": 2.5, "note": "test"})
        assert len(db.get_metrics(run_id)) == 1

    def test_log_metrics_empty(self, db):
        run_id = db.create_run("test", {})
        db.log_metrics(run_id, step=0, metrics={})
        assert db.get_metrics(run_id) == []

    def test_metrics_ordered_by_step(self, db):
        run_id = db.create_run("test", {})
        db.log_metrics(run_id, step=10, metrics={"loss": 1.0})
        db.log_metrics(run_id, step=5, metrics={"loss": 2.0})
        db.log_metrics(run_id, step=20, metrics={"loss": 0.5})

        metrics = db.get_metrics(run_id, key="loss")
        steps = [m["step"] for m in metrics]
        assert steps == [5, 10, 20]


class TestEvals:
    def test_log_and_get_eval(self, db):
        run_id = db.create_run("test", {})
        db.log_eval(
            task="perplexity",
            metrics={"perplexity": 45.2, "bpc": 1.3},
            metadata={"num_tokens": 1000},
            run_id=run_id,
            step=100,
        )

        evals = db.get_evals(run_id=run_id)
        assert len(evals) == 1
        assert evals[0]["task"] == "perplexity"
        assert json.loads(evals[0]["metrics"]) == {"perplexity": 45.2, "bpc": 1.3}

    def test_standalone_eval_no_run(self, db):
        db.log_eval(
            task="blimp",
            metrics={"accuracy": 0.82},
            model_name="smollm-135m",
        )
        evals = db.get_evals()
        assert len(evals) == 1
        assert evals[0]["run_id"] is None
        assert evals[0]["model_name"] == "smollm-135m"

    def test_filter_by_task(self, db):
        db.log_eval(task="perplexity", metrics={"perplexity": 45.0})
        db.log_eval(task="blimp", metrics={"accuracy": 0.8})
        db.log_eval(task="perplexity", metrics={"perplexity": 40.0})

        ppl_evals = db.get_evals(task="perplexity")
        assert len(ppl_evals) == 2

        blimp_evals = db.get_evals(task="blimp")
        assert len(blimp_evals) == 1

    def test_filter_by_run_and_task(self, db):
        r1 = db.create_run("run1", {})
        r2 = db.create_run("run2", {})
        db.log_eval(task="perplexity", metrics={"ppl": 50}, run_id=r1)
        db.log_eval(task="perplexity", metrics={"ppl": 40}, run_id=r2)

        evals = db.get_evals(run_id=r1, task="perplexity")
        assert len(evals) == 1
        assert json.loads(evals[0]["metrics"])["ppl"] == 50


class TestCheckpoints:
    def test_log_and_get_checkpoints(self, db):
        run_id = db.create_run("test", {})
        db.log_checkpoint(run_id, step=100, path="/tmp/ckpt-100.pt", metrics={"val/loss": 2.0})
        db.log_checkpoint(run_id, step=200, path="/tmp/ckpt-200.pt", metrics={"val/loss": 1.5}, is_best=True)

        checkpoints = db.get_checkpoints(run_id)
        assert len(checkpoints) == 2
        assert checkpoints[0]["step"] == 100
        assert checkpoints[0]["is_best"] == 0
        assert checkpoints[1]["step"] == 200
        assert checkpoints[1]["is_best"] == 1

    def test_checkpoint_metrics_json(self, db):
        run_id = db.create_run("test", {})
        db.log_checkpoint(run_id, step=50, path="/tmp/ckpt.pt", metrics={"val/loss": 1.2, "val/ppl": 3.3})

        ckpt = db.get_checkpoints(run_id)[0]
        assert json.loads(ckpt["metrics"]) == {"val/loss": 1.2, "val/ppl": 3.3}


class TestContextManager:
    def test_context_manager(self):
        fd, path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        with Database(path) as db:
            run_id = db.create_run("ctx-test", {})
            db.log_metrics(run_id, 0, {"loss": 1.0})
        # DB should be closed, but file should exist with data
        db2 = Database(path)
        assert len(db2.list_runs()) == 1
        db2.close()
        os.unlink(path)


class TestLoggerIntegration:
    def test_logger_writes_to_db(self, db):
        from src.config.base import LoggingConfig
        from src.logging.logger import Logger

        run_id = db.create_run("logger-test", {})
        config = LoggingConfig(console_interval=1000)  # suppress console
        logger = Logger(config, "test", db=db, run_id=run_id)

        logger.log_step({"train/loss": 2.5, "train/lr": 1e-3}, step=0)
        logger.log_step({"train/loss": 2.0, "train/lr": 9e-4}, step=1)

        metrics = db.get_metrics(run_id, key="train/loss")
        assert len(metrics) == 2
        assert metrics[0]["value"] == 2.5
        assert metrics[1]["value"] == 2.0

    def test_logger_without_db(self):
        from src.config.base import LoggingConfig
        from src.logging.logger import Logger

        config = LoggingConfig(console_interval=1000)
        logger = Logger(config, "test")
        # Should not raise
        logger.log_step({"train/loss": 2.5}, step=0)
        logger.close()
