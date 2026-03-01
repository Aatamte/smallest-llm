"""Tests for RunManager — run lifecycle, stop, stale recovery, shutdown."""

import os
import tempfile
import threading
import time

import pytest

from src.server.broadcast import Broadcaster
from src.server.run_manager import RunManager
from src.storage.database import Database


@pytest.fixture
def db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    os.unlink(path)


@pytest.fixture
def broadcaster():
    return Broadcaster()


@pytest.fixture
def manager(db_path, broadcaster):
    mgr = RunManager(db_path, broadcaster)
    yield mgr
    mgr.db.close()


# ── get_active ────────────────────────────────────────────


class TestGetActive:
    def test_no_active_run_initially(self, manager):
        assert manager.get_active() is None

    def test_returns_active_after_set(self, manager):
        manager._active_run_id = 1
        active = manager.get_active()
        assert active is not None
        assert active["run_id"] == 1
        assert active["status"] == "running"

        # Cleanup
        manager._active_run_id = None


# ── stop ──────────────────────────────────────────────────


class TestStop:
    def test_stop_when_no_run(self, manager):
        assert manager.stop() is False

    def test_stop_sets_should_stop_and_waits(self, manager):
        """Simulate a running trainer and verify stop signals it."""
        run_id = manager.db.create_run("test", {})

        # Fake trainer with should_stop
        class FakeTrainer:
            should_stop = False

        trainer = FakeTrainer()
        stopped_event = threading.Event()

        def fake_run():
            while not trainer.should_stop:
                time.sleep(0.01)
            stopped_event.set()

        thread = threading.Thread(target=fake_run)
        thread.start()

        manager._active_run_id = run_id
        manager._active_trainer = trainer
        manager._active_thread = thread

        ok = manager.stop(timeout=5.0)
        assert ok is True
        assert stopped_event.is_set()
        assert not thread.is_alive()

        # DB should be marked stopped
        run = manager.db.get_run(run_id)
        assert run["status"] == "stopped"

    def test_stop_timeout_marks_failed(self, manager):
        """If the thread doesn't stop in time, it's marked failed."""
        run_id = manager.db.create_run("stuck", {})

        class FakeTrainer:
            should_stop = False

        trainer = FakeTrainer()

        # Thread that ignores should_stop
        def stuck_run():
            time.sleep(10)

        thread = threading.Thread(target=stuck_run, daemon=True)
        thread.start()

        manager._active_run_id = run_id
        manager._active_trainer = trainer
        manager._active_thread = thread

        ok = manager.stop(timeout=0.1)
        assert ok is False

        run = manager.db.get_run(run_id)
        assert run["status"] == "failed"


# ── recover_stale ─────────────────────────────────────────


class TestRecoverStale:
    def test_marks_running_as_failed(self, db_path, broadcaster):
        # Seed runs into the DB before creating the manager
        db = Database(db_path)
        r1 = db.create_run("stale-1", {})
        r2 = db.create_run("stale-2", {})
        r3 = db.create_run("done", {})
        db.finish_run(r3, status="completed")
        db.close()

        mgr = RunManager(db_path, broadcaster)
        mgr.recover_stale()

        assert mgr.db.get_run(r1)["status"] == "failed"
        assert mgr.db.get_run(r2)["status"] == "failed"
        assert mgr.db.get_run(r3)["status"] == "completed"
        mgr.db.close()

    def test_no_stale_runs(self, manager):
        r1 = manager.db.create_run("done", {})
        manager.db.finish_run(r1, status="completed")
        stale = manager.recover_stale()
        assert stale == []

    def test_empty_db(self, manager):
        stale = manager.recover_stale()
        assert stale == []


# ── shutdown ──────────────────────────────────────────────


class TestShutdown:
    def test_shutdown_no_active_run(self, manager):
        # Should not raise
        manager.shutdown()

    def test_shutdown_stops_active_run(self, manager):
        run_id = manager.db.create_run("active", {})

        class FakeTrainer:
            should_stop = False

        trainer = FakeTrainer()

        def fake_run():
            while not trainer.should_stop:
                time.sleep(0.01)

        thread = threading.Thread(target=fake_run)
        thread.start()

        manager._active_run_id = run_id
        manager._active_trainer = trainer
        manager._active_thread = thread

        manager.shutdown()
        assert not thread.is_alive()


# ── Database.mark_stale_runs ──────────────────────────────


# ── Stop race condition (#3) ──────────────────────────────


class TestStopRaceCondition:
    def test_stop_does_not_race_with_thread_completion(self, manager):
        """When stop() is called and the thread finishes naturally at the same time,
        the final DB status should be 'stopped', not 'completed'."""
        run_id = manager.db.create_run("race-test", {})

        class FakeTrainer:
            should_stop = False

        trainer = FakeTrainer()
        thread_started = threading.Event()

        def fake_run():
            thread_started.set()
            # Simulate training that finishes quickly after should_stop
            while not trainer.should_stop:
                time.sleep(0.01)
            # Simulate the _run() completing — in the old code this would
            # call db.finish_run("completed") racing with stop()'s "stopped"

        thread = threading.Thread(target=fake_run)
        thread.start()
        thread_started.wait()

        manager._active_run_id = run_id
        manager._active_trainer = trainer
        manager._active_thread = thread

        ok = manager.stop(timeout=5.0)
        assert ok is True

        # The key assertion: status should be deterministically 'stopped'
        run = manager.db.get_run(run_id)
        assert run["status"] == "stopped"
        assert run["finished_at"] is not None

    def test_stop_flag_prevents_thread_from_setting_completed(self, manager):
        """After stop() sets _stopped flag, the training thread should NOT
        overwrite the DB status to 'completed'."""
        run_id = manager.db.create_run("flag-test", {})
        statuses_written = []

        # Patch finish_run to track calls
        original_finish = manager.db.finish_run

        def tracking_finish(rid, status="completed"):
            statuses_written.append(status)
            original_finish(rid, status=status)

        manager.db.finish_run = tracking_finish

        class FakeTrainer:
            should_stop = False

        trainer = FakeTrainer()
        train_done = threading.Event()

        def fake_run():
            while not trainer.should_stop:
                time.sleep(0.01)
            # Simulate what _run() does: try to mark completed
            if not manager._stop_requested:
                manager.db.finish_run(run_id, status="completed")
            train_done.set()

        thread = threading.Thread(target=fake_run)
        thread.start()

        manager._active_run_id = run_id
        manager._active_trainer = trainer
        manager._active_thread = thread

        ok = manager.stop(timeout=5.0)
        train_done.wait(timeout=2.0)
        assert ok is True

        # 'completed' should NOT appear — only 'stopped'
        run = manager.db.get_run(run_id)
        assert run["status"] == "stopped"

        manager.db.finish_run = original_finish


# ── Concurrent DB access (#4) ────────────────────────────


class TestConcurrentDBAccess:
    def test_concurrent_writes_from_multiple_threads(self, manager):
        """Multiple threads writing to the DB concurrently should not corrupt data."""
        run_id = manager.db.create_run("concurrent-test", {})
        errors = []
        num_threads = 4
        writes_per_thread = 50

        def writer(thread_id):
            try:
                for i in range(writes_per_thread):
                    manager.db.log_metrics(
                        run_id,
                        step=thread_id * 1000 + i,
                        metrics={"train/loss": float(i), "train/lr": 0.001},
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert errors == [], f"Concurrent write errors: {errors}"

        # Verify all writes landed
        metrics = manager.db.get_metrics(run_id, key="train/loss")
        assert len(metrics) == num_threads * writes_per_thread

    def test_concurrent_read_write(self, manager):
        """Reads and writes happening concurrently should not error."""
        run_id = manager.db.create_run("rw-test", {})
        errors = []
        stop = threading.Event()

        def writer():
            try:
                for i in range(100):
                    manager.db.log_metrics(run_id, step=i, metrics={"loss": float(i)})
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("write", e))
            finally:
                stop.set()

        def reader():
            try:
                while not stop.is_set():
                    manager.db.get_metrics(run_id)
                    manager.db.get_run(run_id)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(("read", e))

        wt = threading.Thread(target=writer)
        rt = threading.Thread(target=reader)
        wt.start()
        rt.start()
        wt.join(timeout=10.0)
        rt.join(timeout=10.0)

        assert errors == [], f"Concurrent read/write errors: {errors}"

    def test_concurrent_finish_run(self, manager):
        """Two threads calling finish_run on the same run should not crash."""
        run_id = manager.db.create_run("double-finish", {})
        errors = []

        def finish(status):
            try:
                manager.db.finish_run(run_id, status=status)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=finish, args=("completed",))
        t2 = threading.Thread(target=finish, args=("stopped",))
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert errors == [], f"Concurrent finish errors: {errors}"
        # One of the two statuses should have won
        run = manager.db.get_run(run_id)
        assert run["status"] in ("completed", "stopped")


# ── Database.mark_stale_runs ──────────────────────────────


class TestMarkStaleRuns:
    def test_marks_stale(self, db_path):
        db = Database(db_path)
        r1 = db.create_run("a", {})
        r2 = db.create_run("b", {})
        db.finish_run(r2, status="completed")
        r3 = db.create_run("c", {})

        stale = db.mark_stale_runs()
        assert set(stale) == {r1, r3}

        assert db.get_run(r1)["status"] == "failed"
        assert db.get_run(r1)["finished_at"] is not None
        assert db.get_run(r2)["status"] == "completed"
        assert db.get_run(r3)["status"] == "failed"
        db.close()

    def test_no_stale(self, db_path):
        db = Database(db_path)
        r1 = db.create_run("done", {})
        db.finish_run(r1, status="completed")
        assert db.mark_stale_runs() == []
        db.close()

    def test_empty(self, db_path):
        db = Database(db_path)
        assert db.mark_stale_runs() == []
        db.close()
