"""FastAPI backend for the training dashboard."""

from __future__ import annotations

import asyncio
import json
import math
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.server.broadcast import broadcaster
from src.server.run_manager import RunManager
from src.storage.eval_db import EvalDatabase
from src.types.status import db_to_wire

# ── Shared state ─────────────────────────────────────────

DB_PATH = "smallest_llm.db"
EVAL_DB_PATH = "eval.db"
run_manager = RunManager(DB_PATH, broadcaster)
eval_db = EvalDatabase(EVAL_DB_PATH)


@asynccontextmanager
async def lifespan(app: FastAPI):
    broadcaster.set_loop(asyncio.get_running_loop())
    run_manager.recover_stale()
    yield
    run_manager.shutdown()
    eval_db.close()


app = FastAPI(title="smallest-llm dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST endpoints ────────────────────────────────────────

@app.get("/api/runs")
def list_runs():
    return run_manager.db.list_runs()


@app.get("/api/runs/{run_id}")
def get_run(run_id: int):
    run = run_manager.db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    run["config"] = json.loads(run["config"]) if run.get("config") else {}
    run["env"] = json.loads(run["env"]) if run.get("env") else {}
    return run


@app.get("/api/runs/{run_id}/metrics")
def get_metrics(run_id: int, key: Optional[str] = Query(None)):
    return run_manager.db.get_metrics(run_id, key=key)


@app.get("/api/runs/{run_id}/evals")
def get_evals(run_id: int):
    evals = eval_db.get_evals(run_id=run_id)
    for e in evals:
        e["metrics"] = json.loads(e["metrics"]) if e.get("metrics") else {}
        e["metadata"] = json.loads(e["metadata"]) if e.get("metadata") else {}
    return evals


@app.get("/api/evals")
def list_all_evals(model_name: Optional[str] = Query(None), task: Optional[str] = Query(None)):
    """List eval results, optionally filtered by model_name or task."""
    evals = eval_db.get_evals(model_name=model_name, task=task)
    for e in evals:
        e["metrics"] = json.loads(e["metrics"]) if e.get("metrics") else {}
        e["metadata"] = json.loads(e["metadata"]) if e.get("metadata") else {}
    return evals


@app.get("/api/evals/models")
def list_eval_models():
    """List distinct model names that have eval results."""
    return eval_db.list_models()


@app.get("/api/runs/{run_id}/checkpoints")
def get_checkpoints(run_id: int):
    checkpoints = run_manager.db.get_checkpoints(run_id)
    for c in checkpoints:
        c["metrics"] = json.loads(c["metrics"]) if c.get("metrics") else {}
    return checkpoints


@app.get("/api/runs/{run_id}/checkpoints/{step}/weights")
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


@app.get("/api/runs/{run_id}/state")
def get_training_state(run_id: int):
    """Full TrainingState snapshot matching the dashboard's expected shape."""
    run = run_manager.db.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")

    config = json.loads(run["config"]) if run.get("config") else {}
    raw_metrics = run_manager.db.get_metrics(run_id)

    # Group metrics by step
    steps_dict: dict[int, dict[str, float]] = {}
    for m in raw_metrics:
        step = m["step"]
        if step not in steps_dict:
            steps_dict[step] = {}
        steps_dict[step][m["key"]] = m["value"]

    # Transform to StepMetrics[]
    steps = []
    for step in sorted(steps_dict.keys()):
        sm = steps_dict[step]
        step_metrics = {
            "step": step,
            "trainLoss": sm.get("train/loss", 0),
            "lr": sm.get("train/lr", 0),
            "gradNorm": sm.get("train/grad_norm", 0),
            "tokensSeen": int(sm.get("train/tokens_seen", 0)),
            "tokensPerSec": _tokens_per_sec(sm),
            "stepTime": sm.get("train/step_time", 0),
        }
        if "val/loss" in sm:
            step_metrics["valLoss"] = sm["val/loss"]
        if "train/loss" in sm:
            step_metrics["bpc"] = sm["train/loss"] / math.log(2)
        steps.append(step_metrics)

    latest = steps[-1] if steps else {}
    best_val = float("inf")
    for s in steps:
        if "valLoss" in s and s["valLoss"] < best_val:
            best_val = s["valLoss"]

    training_config = config.get("training", {})

    return {
        "experimentName": run["name"],
        "status": db_to_wire(run["status"]),
        "maxSteps": training_config.get("max_steps", 10000),
        "startTime": run.get("created_at", ""),
        "steps": steps,
        "currentStep": latest.get("step", 0),
        "currentTrainLoss": latest.get("trainLoss", 0),
        "currentValLoss": latest.get("valLoss", 0),
        "bestValLoss": best_val if best_val < float("inf") else 0,
        "currentLR": latest.get("lr", 0),
        "currentGradNorm": latest.get("gradNorm", 0),
        "tokensSeen": latest.get("tokensSeen", 0),
        "tokensPerSec": latest.get("tokensPerSec", 0),
        "bpc": latest.get("bpc", 0),
        "layerStats": [],
        "generations": [],
    }


@app.get("/api/config")
def get_config():
    """Return the default experiment config."""
    from src.config.base import ExperimentConfig
    return ExperimentConfig().to_dict()


# ── Presets ────────────────────────────────────────────────

@app.get("/api/presets")
def list_presets():
    """Return list of available config presets."""
    from src.config.presets import get_presets
    return get_presets()


@app.get("/api/presets/{name}")
def get_preset_config(name: str):
    """Return full config dict for a named preset."""
    from src.config.presets import get_preset
    preset = get_preset(name)
    if preset is None:
        raise HTTPException(status_code=404, detail=f"Preset '{name}' not found")
    return preset.to_dict()


# ── Training control ──────────────────────────────────────

@app.get("/api/active-run")
def get_active_run():
    """Return the currently active run, or null."""
    return run_manager.get_active()


@app.post("/api/runs/start")
async def start_training_run(request: Request):
    """Start a training run, optionally with a custom config."""
    from src.config.base import ExperimentConfig

    body = await request.body()
    config = None
    if body:
        data = json.loads(body)
        config = ExperimentConfig.from_dict(data)

    try:
        run_id = run_manager.start(config=config)
        return {"status": "started", "run_id": run_id}
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@app.post("/api/runs/{run_id}/stop")
def stop_training_run(run_id: int):
    """Gracefully stop a training run by ID."""
    try:
        ok = run_manager.stop(run_id)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return {"status": "stopped" if ok else "failed", "run_id": run_id}


# ── WebSocket ─────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        async for message in broadcaster.subscribe():
            await ws.send_text(message)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass


# ── Helpers ───────────────────────────────────────────────

def _tokens_per_sec(sm: dict[str, float]) -> float:
    step_time = sm.get("train/step_time", 0)
    tokens = sm.get("train/tokens_seen", 0)
    if step_time > 0:
        return tokens / step_time if tokens > 0 else 0
    return 0


