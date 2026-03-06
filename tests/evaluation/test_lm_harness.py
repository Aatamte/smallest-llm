"""Tests for lm-evaluation-harness integration."""

import json
import os
import tempfile

import pytest

from src.evaluation.lm_harness_results import (
    harness_results_to_eval_results,
    persist_harness_results,
)
from src.storage import EvalDatabase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eval_db():
    """Create a temporary eval database for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = EvalDatabase(path)
    yield db
    db.close()
    os.unlink(path)


@pytest.fixture
def mock_harness_output():
    """Simulated lm-eval-harness output from simple_evaluate()."""
    return {
        "results": {
            "hellaswag": {
                "acc,none": 0.2531,
                "acc_norm,none": 0.2789,
                "acc_stderr,none": 0.0043,
                "acc_norm_stderr,none": 0.0045,
                "alias": "hellaswag",
            },
            "arc_easy": {
                "acc,none": 0.3512,
                "acc_norm,none": 0.3201,
                "acc_stderr,none": 0.0098,
                "acc_norm_stderr,none": 0.0096,
                "alias": "arc_easy",
            },
        },
        "config": {
            "num_fewshot": 0,
            "batch_size": 4,
            "limit": 100,
        },
    }


# ---------------------------------------------------------------------------
# Result mapping tests
# ---------------------------------------------------------------------------


class TestHarnessResultMapping:
    def test_converts_tasks(self, mock_harness_output):
        results = harness_results_to_eval_results(mock_harness_output)
        assert "harness/hellaswag" in results
        assert "harness/arc_easy" in results

    def test_prefixes_task_names(self, mock_harness_output):
        results = harness_results_to_eval_results(mock_harness_output)
        for task_name in results:
            assert task_name.startswith("harness/")

    def test_cleans_metric_keys(self, mock_harness_output):
        results = harness_results_to_eval_results(mock_harness_output)
        hs = results["harness/hellaswag"]
        # "acc,none" should become "acc"
        assert "acc" in hs.metrics
        assert "acc_norm" in hs.metrics
        # Raw keys should not be present
        assert "acc,none" not in hs.metrics

    def test_metrics_are_rounded(self, mock_harness_output):
        results = harness_results_to_eval_results(mock_harness_output)
        hs = results["harness/hellaswag"]
        assert hs.metrics["acc"] == 0.2531
        assert hs.metrics["acc_norm"] == 0.2789

    def test_stderr_in_metadata(self, mock_harness_output):
        results = harness_results_to_eval_results(mock_harness_output)
        hs = results["harness/hellaswag"]
        assert "acc_stderr" in hs.metadata
        assert hs.metadata["acc_stderr"] == 0.0043
        # stderr should NOT be in metrics
        assert "acc_stderr" not in hs.metrics

    def test_alias_in_metadata(self, mock_harness_output):
        results = harness_results_to_eval_results(mock_harness_output)
        hs = results["harness/hellaswag"]
        assert hs.metadata["alias"] == "hellaswag"

    def test_config_in_metadata(self, mock_harness_output):
        results = harness_results_to_eval_results(mock_harness_output)
        hs = results["harness/hellaswag"]
        assert hs.metadata["num_fewshot"] == 0
        assert hs.metadata["batch_size"] == 4
        assert hs.metadata["limit"] == 100

    def test_source_tag(self, mock_harness_output):
        results = harness_results_to_eval_results(mock_harness_output)
        for result in results.values():
            assert result.metadata["source"] == "lm-evaluation-harness"

    def test_empty_results(self):
        results = harness_results_to_eval_results({"results": {}})
        assert results == {}

    def test_nan_values_excluded(self):
        output = {
            "results": {
                "test_task": {
                    "acc,none": float("nan"),
                    "acc_norm,none": 0.5,
                },
            },
        }
        results = harness_results_to_eval_results(output)
        task = results["harness/test_task"]
        assert "acc" not in task.metrics
        assert "acc_norm" in task.metrics

    def test_summary_line(self, mock_harness_output):
        results = harness_results_to_eval_results(mock_harness_output)
        line = results["harness/hellaswag"].summary_line()
        assert "harness/hellaswag" in line
        assert "acc" in line


# ---------------------------------------------------------------------------
# DB persistence tests
# ---------------------------------------------------------------------------


class TestHarnessPersistence:
    def test_persist_and_retrieve(self, eval_db, mock_harness_output):
        results = harness_results_to_eval_results(mock_harness_output)
        persist_harness_results(results, eval_db, model_name="test-model")

        evals = eval_db.get_evals(model_name="test-model")
        assert len(evals) == 2

        tasks = {e["task"] for e in evals}
        assert "harness/hellaswag" in tasks
        assert "harness/arc_easy" in tasks

    def test_persist_metrics_correct(self, eval_db, mock_harness_output):
        results = harness_results_to_eval_results(mock_harness_output)
        persist_harness_results(results, eval_db, model_name="test-model")

        evals = eval_db.get_evals(task="harness/hellaswag")
        assert len(evals) == 1
        metrics = json.loads(evals[0]["metrics"])
        assert metrics["acc"] == 0.2531
        assert metrics["acc_norm"] == 0.2789

    def test_persist_with_run_id(self, eval_db, mock_harness_output):
        results = harness_results_to_eval_results(mock_harness_output)
        persist_harness_results(
            results, eval_db, model_name="ckpt", run_id=42, step=500
        )

        evals = eval_db.get_evals(model_name="ckpt")
        for e in evals:
            assert e["run_id"] == 42
            assert e["step"] == 500

    def test_persist_empty_results(self, eval_db):
        persist_harness_results({}, eval_db, model_name="empty")
        evals = eval_db.get_evals(model_name="empty")
        assert len(evals) == 0

    def test_model_appears_in_list(self, eval_db, mock_harness_output):
        results = harness_results_to_eval_results(mock_harness_output)
        persist_harness_results(results, eval_db, model_name="my-model")

        models = eval_db.list_models()
        assert "my-model" in models
