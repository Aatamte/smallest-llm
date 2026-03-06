"""Run CRUD and training control endpoints."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from src.server.state import eval_db, run_manager
from src.types.checkpoint import CheckpointRecord
from src.types.eval import EvalRecord
from src.types.run import RunRecord
from src.types.status import db_to_wire
from src.types.training_state import TrainingStateSnapshot

router = APIRouter()


@router.get("/api/runs")
def list_runs():
    return run_manager.db.list_runs()


@router.get("/api/runs/{run_id}")
def get_run(run_id: int):
    run = run_manager.db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return RunRecord.from_db_row(run).to_wire()


@router.get("/api/runs/{run_id}/metrics")
def get_metrics(run_id: int, key: Optional[str] = Query(None)):
    return run_manager.db.get_metrics(run_id, key=key)


@router.get("/api/runs/{run_id}/evals")
def get_evals(run_id: int):
    rows = eval_db.get_evals(run_id=run_id)
    return [EvalRecord.from_db_row(r).to_wire() for r in rows]


@router.get("/api/runs/{run_id}/checkpoints")
def get_checkpoints(run_id: int):
    rows = run_manager.db.get_checkpoints(run_id)
    return [CheckpointRecord.from_db_row(r).to_wire() for r in rows]


@router.get("/api/runs/{run_id}/checkpoints/{step}/weights")
def get_checkpoint_weights(run_id: int, step: int):
    """Return downsampled weight tensors from a checkpoint."""
    import os
    from src.server.weights import extract_weights

    checkpoints = run_manager.db.get_checkpoints(run_id)
    match = next((c for c in checkpoints if c["step"] == step), None)
    if match is None:
        raise HTTPException(status_code=404, detail=f"No checkpoint at step {step}")

    path = match["path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Checkpoint file not found: {path}")

    layers = extract_weights(path)
    return {"layers": layers}


@router.get("/api/runs/{run_id}/state")
def get_training_state(run_id: int):
    """Full TrainingState snapshot matching the dashboard's expected shape."""
    run = run_manager.db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    config = json.loads(run["config"]) if run.get("config") else {}
    raw_metrics = run_manager.db.get_metrics(run_id)

    steps_dict: dict[int, dict[str, float]] = {}
    for m in raw_metrics:
        step = m["step"]
        if step not in steps_dict:
            steps_dict[step] = {}
        steps_dict[step][m["key"]] = m["value"]

    series = TrainingStateSnapshot.build_series(steps_dict)
    training_config = config.get("training", {})

    snapshot = TrainingStateSnapshot(
        experiment_name=run["name"],
        status=db_to_wire(run["status"]),
        text_state=run_manager.broadcaster.text_state if run["status"] == "running" else "",
        max_steps=training_config.get("max_steps", 10000),
        start_time=run.get("created_at", ""),
        series=series,
        generations=[],
    )
    return snapshot.to_wire()


@router.get("/api/active-run")
def get_active_run():
    """Return the currently active run, or null."""
    return run_manager.get_active()


@router.post("/api/runs/start")
async def start_training_run(request: Request):
    """Start a training run, optionally with a custom config."""
    from src.config.base import ExperimentConfig
    from src.evaluation.config import get_eval_preset

    body = await request.body()
    config = None
    if body:
        data = json.loads(body)
        # Apply eval preset if specified
        eval_preset_name = data.pop("eval_preset", None)
        # Apply FLOPs budget override if specified
        flops_budget_name = data.pop("flops_budget", None)
        config = ExperimentConfig.from_dict(data)
        if eval_preset_name:
            eval_preset = get_eval_preset(eval_preset_name)
            if eval_preset:
                config.training.eval_tasks = len(eval_preset.tasks) > 0
                config.training.eval_tasks_list = ",".join(eval_preset.tasks)
                config.training.eval_tasks_interval = eval_preset.interval
                config.training.eval_tasks_max_samples = eval_preset.max_samples or 0
        if flops_budget_name:
            from src.config.presets import get_flops_budget
            budget = get_flops_budget(flops_budget_name)
            if budget is not None:
                # Scale stage budgets proportionally if stages exist
                old_budget = config.training.max_flops
                config.training.max_flops = budget
                config.training.max_steps = 0
                if config.stages and old_budget and old_budget > 0:
                    ratio = budget / old_budget
                    for stage in config.stages:
                        if stage.max_flops is not None:
                            stage.max_flops *= ratio

    try:
        run_id = run_manager.start(config=config)
        return {"status": "started", "run_id": run_id}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/api/runs/{run_id}/stop")
def stop_training_run(run_id: int):
    """Gracefully stop a training run by ID."""
    try:
        ok = run_manager.stop(run_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"status": "stopped" if ok else "failed", "run_id": run_id}


@router.delete("/api/runs/{run_id}")
def delete_run(run_id: int):
    """Delete a run and all its associated data."""
    run_manager.delete(run_id)
    eval_db.delete_by_run(run_id)
    return {"status": "deleted", "run_id": run_id}


@router.post("/api/runs/bulk-delete")
async def bulk_delete_runs(request: Request):
    """Delete multiple runs at once. Body: {"run_ids": [1, 2, 3]}"""
    body = json.loads(await request.body())
    run_ids = body.get("run_ids", [])
    deleted = []
    for rid in run_ids:
        run_manager.delete(rid)
        eval_db.delete_by_run(rid)
        deleted.append(rid)
    return {"status": "deleted", "run_ids": deleted}
