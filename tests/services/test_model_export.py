"""End-to-end test: training completes → model exported → model in DB → API serves it."""

import os
import shutil
import tempfile
import time

import pytest

from src.config.base import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    StageConfig,
    TrainingConfig,
)
from src.server.broadcast import Broadcaster
from src.services.run_service import RunManager


def _tiny_config(name="test-export", stages=None):
    """Minimal config that trains in <1 second."""
    return ExperimentConfig(
        name=name,
        device="cpu",
        model=ModelConfig(name="transformer", extra_args={"d_model": 16, "n_heads": 1, "n_layers": 1}),
        data=DataConfig(max_seq_len=8, batch_size=2),
        training=TrainingConfig(max_steps=5, eval_interval=5, log_interval=5, save_interval=5),
        optimizer=OptimizerConfig(lr=1e-3),
        scheduler=SchedulerConfig(warmup_steps=1),
        stages=stages,
    )


@pytest.fixture
def work_dir(tmp_path, monkeypatch):
    """Run tests in a temp directory so exported models don't pollute the repo."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def manager(work_dir):
    db_path = str(work_dir / "test.db")
    broadcaster = Broadcaster()
    mgr = RunManager(db_path, broadcaster)
    yield mgr
    mgr.db.close()


def _wait_for_run(manager, timeout=30):
    """Wait for the active run thread to finish."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if manager._active_thread is None or not manager._active_thread.is_alive():
            return True
        time.sleep(0.1)
    return False


class TestSingleStageExport:
    def test_training_exports_model_to_disk(self, manager):
        config = _tiny_config()
        run_id = manager.start(config)
        assert _wait_for_run(manager), "Training did not finish in time"

        # Model directory should exist with expected files
        run = manager.db.get_run(run_id)
        model_dir = os.path.join("models", run["name"])
        assert os.path.isdir(model_dir)
        assert os.path.isfile(os.path.join(model_dir, "model.safetensors"))
        assert os.path.isfile(os.path.join(model_dir, "config.json"))
        assert os.path.isfile(os.path.join(model_dir, "tokenizer.json"))
        assert os.path.isfile(os.path.join(model_dir, "experiment_config.json"))

    def test_training_creates_model_in_db(self, manager):
        config = _tiny_config()
        run_id = manager.start(config)
        assert _wait_for_run(manager), "Training did not finish in time"

        models = manager.db.list_models()
        assert len(models) == 1
        assert models[0]["run_id"] == run_id
        assert models[0]["name"] is not None
        assert models[0]["path"].startswith("models/")

    def test_run_marked_completed(self, manager):
        config = _tiny_config()
        run_id = manager.start(config)
        assert _wait_for_run(manager), "Training did not finish in time"

        run = manager.db.get_run(run_id)
        assert run["status"] == "completed"


class TestPipelineExport:
    def test_pipeline_exports_model_to_disk(self, manager):
        config = _tiny_config(
            name="test-pipeline-export",
            stages=[
                StageConfig(name="a", max_steps=3, seq_len=8, eval_interval=3, log_interval=3, save_interval=3),
                StageConfig(name="b", max_steps=3, seq_len=8, eval_interval=3, log_interval=3, save_interval=3),
            ],
        )
        run_id = manager.start(config)
        assert _wait_for_run(manager), "Pipeline did not finish in time"

        run = manager.db.get_run(run_id)
        model_dir = os.path.join("models", run["name"])
        assert os.path.isdir(model_dir)
        assert os.path.isfile(os.path.join(model_dir, "model.safetensors"))
        assert os.path.isfile(os.path.join(model_dir, "config.json"))

    def test_pipeline_creates_model_in_db(self, manager):
        config = _tiny_config(
            name="test-pipeline-db",
            stages=[
                StageConfig(name="a", max_steps=3, seq_len=8, eval_interval=3, log_interval=3, save_interval=3),
                StageConfig(name="b", max_steps=3, seq_len=8, eval_interval=3, log_interval=3, save_interval=3),
            ],
        )
        run_id = manager.start(config)
        assert _wait_for_run(manager), "Pipeline did not finish in time"

        models = manager.db.list_models()
        assert len(models) == 1
        assert models[0]["run_id"] == run_id

    def test_pipeline_run_marked_completed(self, manager):
        config = _tiny_config(
            name="test-pipeline-status",
            stages=[
                StageConfig(name="a", max_steps=3, seq_len=8, eval_interval=3, log_interval=3, save_interval=3),
            ],
        )
        run_id = manager.start(config)
        assert _wait_for_run(manager), "Pipeline did not finish in time"

        run = manager.db.get_run(run_id)
        assert run["status"] == "completed"


class TestStoppedRunSkipsExport:
    def test_stopped_run_does_not_export(self, manager):
        config = _tiny_config()
        config.training.max_steps = 500  # long enough to stop mid-run
        run_id = manager.start(config)

        time.sleep(0.3)
        manager.stop(run_id, timeout=10.0)

        models = manager.db.list_models()
        assert len(models) == 0, "Stopped run should not export a model"


class TestApiServesModels:
    def test_list_models_api(self, manager):
        config = _tiny_config()
        run_id = manager.start(config)
        assert _wait_for_run(manager), "Training did not finish in time"

        # Simulate what the API endpoint does: run_manager.db.list_models()
        models = manager.db.list_models()
        assert len(models) >= 1
        m = models[0]
        assert "id" in m
        assert "run_id" in m
        assert "name" in m
        assert "path" in m
        assert "created_at" in m
