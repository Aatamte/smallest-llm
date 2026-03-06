"""Tests for EvalDatabase (eval.db)."""

import os
import tempfile

import pytest

from src.storage import EvalDatabase


@pytest.fixture
def db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = EvalDatabase(path)
    yield database
    database.close()
    os.unlink(path)


class TestLogEval:
    def test_log_and_get(self, db):
        db.log_eval("piqa", {"accuracy": 0.46}, run_id=1, step=2000)
        rows = db.get_evals(run_id=1)
        assert len(rows) == 1
        assert rows[0]["task"] == "piqa"
        assert rows[0]["step"] == 2000

    def test_log_with_metadata(self, db):
        db.log_eval("hellaswag", {"accuracy": 0.23}, metadata={"samples": 100})
        rows = db.get_evals()
        assert rows[0]["metadata"] is not None

    def test_log_with_model_name(self, db):
        db.log_eval("arc_easy", {"accuracy": 0.22}, model_name="smollm-135m")
        rows = db.get_evals(model_name="smollm-135m")
        assert len(rows) == 1

    def test_log_minimal(self, db):
        db.log_eval("perplexity", {"perplexity": 45.2})
        rows = db.get_evals()
        assert len(rows) == 1
        assert rows[0]["run_id"] is None
        assert rows[0]["step"] is None
        assert rows[0]["model_name"] is None


class TestGetEvals:
    def test_filter_by_run_id(self, db):
        db.log_eval("a", {"x": 1}, run_id=1)
        db.log_eval("b", {"x": 2}, run_id=2)
        rows = db.get_evals(run_id=1)
        assert len(rows) == 1
        assert rows[0]["task"] == "a"

    def test_filter_by_task(self, db):
        db.log_eval("piqa", {"accuracy": 0.5})
        db.log_eval("hellaswag", {"accuracy": 0.3})
        rows = db.get_evals(task="piqa")
        assert len(rows) == 1

    def test_filter_by_model_name(self, db):
        db.log_eval("piqa", {"accuracy": 0.5}, model_name="a")
        db.log_eval("piqa", {"accuracy": 0.6}, model_name="b")
        rows = db.get_evals(model_name="b")
        assert len(rows) == 1
        assert rows[0]["metrics"] == '{"accuracy": 0.6}'

    def test_combined_filters(self, db):
        db.log_eval("piqa", {"accuracy": 0.5}, run_id=1, model_name="m")
        db.log_eval("piqa", {"accuracy": 0.6}, run_id=2, model_name="m")
        db.log_eval("arc", {"accuracy": 0.3}, run_id=1, model_name="m")
        rows = db.get_evals(run_id=1, task="piqa")
        assert len(rows) == 1

    def test_no_filters_returns_all(self, db):
        db.log_eval("a", {"x": 1})
        db.log_eval("b", {"x": 2})
        db.log_eval("c", {"x": 3})
        rows = db.get_evals()
        assert len(rows) == 3

    def test_empty(self, db):
        assert db.get_evals() == []


class TestListModels:
    def test_list_distinct_models(self, db):
        db.log_eval("a", {"x": 1}, model_name="model-1")
        db.log_eval("b", {"x": 2}, model_name="model-1")
        db.log_eval("c", {"x": 3}, model_name="model-2")
        models = db.list_models()
        assert models == ["model-1", "model-2"]

    def test_excludes_null_model_names(self, db):
        db.log_eval("a", {"x": 1}, model_name=None)
        db.log_eval("b", {"x": 2}, model_name="real")
        models = db.list_models()
        assert models == ["real"]

    def test_empty(self, db):
        assert db.list_models() == []


class TestDeleteByRun:
    def test_deletes_matching(self, db):
        db.log_eval("a", {"x": 1}, run_id=1)
        db.log_eval("b", {"x": 2}, run_id=1)
        db.log_eval("c", {"x": 3}, run_id=2)
        db.delete_by_run(1)
        rows = db.get_evals()
        assert len(rows) == 1
        assert rows[0]["task"] == "c"

    def test_delete_nonexistent_run_is_noop(self, db):
        db.log_eval("a", {"x": 1}, run_id=1)
        db.delete_by_run(999)
        assert len(db.get_evals()) == 1
