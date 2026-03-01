"""FastAPI backend for the training dashboard."""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
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
CHECKPOINT_DB_PATH = "checkpoints.db"
run_manager = RunManager(DB_PATH, broadcaster, checkpoint_db_path=CHECKPOINT_DB_PATH)
eval_db = EvalDatabase(EVAL_DB_PATH)

# Known HF models available for evaluation
EVAL_MODELS = {
    "smollm-135m": "HuggingFaceTB/SmolLM-135M",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
}

# Eval job state (simple dict, protected by a lock)
_eval_lock = threading.Lock()
_eval_stop = threading.Event()
_eval_state: dict = {
    "status": "idle",
    "model_name": None,
    "task": None,
    "task_index": 0,
    "task_count": 0,
    "current_sample": 0,
    "total_samples": 0,
    "started_at": None,
    "error": None,
}

# Chat model state
_chat_lock = threading.Lock()
_chat_state: dict = {"model": None, "tokenizer": None, "source": None, "name": None}


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
    checkpoints = run_manager.checkpoint_db.get_checkpoints(run_id)
    for c in checkpoints:
        c["metrics"] = json.loads(c["metrics"]) if c.get("metrics") else {}
    return checkpoints


@app.get("/api/runs/{run_id}/checkpoints/{step}/weights")
def get_checkpoint_weights(run_id: int, step: int):
    """Return downsampled weight tensors from a checkpoint."""
    import os
    from src.server.weights import extract_weights

    checkpoints = run_manager.checkpoint_db.get_checkpoints(run_id)
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


@app.delete("/api/runs/{run_id}")
def delete_run(run_id: int):
    """Delete a run and all its associated data."""
    run_manager.delete(run_id)
    eval_db.delete_by_run(run_id)
    return {"status": "deleted", "run_id": run_id}


@app.post("/api/runs/bulk-delete")
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


# ── HF Model Evaluation ──────────────────────────────────

@app.get("/api/evals/available-models")
def list_available_models():
    """Return HF models that can be evaluated."""
    return [{"name": name, "hf_id": hf_id} for name, hf_id in EVAL_MODELS.items()]


@app.get("/api/evals/status")
def get_eval_status():
    """Return current eval job status."""
    with _eval_lock:
        return dict(_eval_state)


@app.post("/api/evals/stop")
def stop_eval():
    """Stop a running evaluation."""
    _eval_stop.set()
    with _eval_lock:
        if _eval_state["status"] == "running":
            _eval_state.update(status="stopped", error=None)
        return dict(_eval_state)


@app.post("/api/evals/run")
async def run_eval(request: Request):
    """Start a model evaluation in a background thread.

    Body for HF model:      {"model_name": "smollm-135m", "tasks": [...]}
    Body for checkpoint:     {"source": "checkpoint", "run_id": 24, "step": 500, "tasks": [...]}
    """
    body = json.loads(await request.body())
    source = body.get("source", "hf")
    tasks = body.get("tasks", ["perplexity", "blimp"])
    max_samples = body.get("max_samples")

    from src.evaluation.tasks import list_tasks
    available_tasks = list_tasks()
    invalid = [t for t in tasks if t not in available_tasks]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown tasks: {invalid}. Available: {available_tasks}",
        )

    if source == "checkpoint":
        run_id = body.get("run_id")
        step = body.get("step")
        if run_id is None or step is None:
            raise HTTPException(status_code=400, detail="run_id and step required for checkpoint eval")

        # Resolve checkpoint path and display name
        checkpoints = run_manager.checkpoint_db.get_checkpoints(run_id)
        match = next((c for c in checkpoints if c["step"] == step), None)
        if match is None:
            raise HTTPException(status_code=404, detail=f"No checkpoint at step {step} for run {run_id}")

        run = run_manager.db.get_run(run_id)
        run_name = run["name"] if run else f"run-{run_id}"
        display_name = f"{run_name} (step {step})"
        checkpoint_path = match["path"]

    elif source == "hf":
        model_name = body.get("model_name")
        if model_name not in EVAL_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown model: {model_name}. Available: {list(EVAL_MODELS.keys())}",
            )
        display_name = model_name

    else:
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}")

    with _eval_lock:
        if _eval_state["status"] == "running":
            raise HTTPException(status_code=409, detail="An evaluation is already running")
        _eval_state.update(
            status="running", model_name=display_name, task=None,
            task_index=0, task_count=len(tasks),
            current_sample=0, total_samples=0,
            started_at=time.time(), error=None,
        )

    def _progress_cb(task_index, task_count, task_name, current, total):
        if _eval_stop.is_set():
            from src.evaluation.runner import EvalCancelled
            raise EvalCancelled("Evaluation stopped by user")
        with _eval_lock:
            _eval_state.update(
                task=task_name, task_index=task_index, task_count=task_count,
                current_sample=current, total_samples=total,
            )

    def _run():
        _eval_stop.clear()
        try:
            from src.evaluation import EvalConfig, evaluate

            if source == "checkpoint":
                from src.evaluation.checkpoint_model import CheckpointModel
                model = CheckpointModel(checkpoint_path)
            else:
                from src.evaluation.hf_model import HFModel
                model = HFModel(EVAL_MODELS[model_name])

            config = EvalConfig(tasks=tasks, max_samples=max_samples)
            evaluate(model, config, db=eval_db, model_name=display_name, on_progress=_progress_cb)

            del model

            with _eval_lock:
                _eval_state.update(
                    status="idle", model_name=None, task=None,
                    task_index=0, task_count=0,
                    current_sample=0, total_samples=0,
                    started_at=None,
                )

        except Exception as e:
            from src.evaluation.runner import EvalCancelled
            if isinstance(e, EvalCancelled):
                with _eval_lock:
                    _eval_state.update(
                        status="stopped", model_name=None, task=None,
                        task_index=0, task_count=0,
                        current_sample=0, total_samples=0,
                        started_at=None, error=None,
                    )
            else:
                import traceback
                traceback.print_exc()
                with _eval_lock:
                    _eval_state.update(status="error", error=str(e))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return {"status": "started", "model_name": display_name, "tasks": tasks}


# ── Chat / Generation ────────────────────────────────────

def _load_checkpoint_model(run_id: int, step: int):
    """Load a trained model + tokenizer from a checkpoint."""
    import os
    import torch
    from src.config.base import ExperimentConfig
    from src.data.tokenizer import build_tokenizer
    from src.data.datasets import load_dataset, TextFileDataset
    from src.training.run import _build_model
    from src.utils.device import resolve_device

    checkpoints = run_manager.checkpoint_db.get_checkpoints(run_id)
    match = next((c for c in checkpoints if c["step"] == step), None)
    if match is None:
        raise ValueError(f"No checkpoint at step {step} for run {run_id}")

    path = match["path"]
    if not os.path.exists(path):
        raise ValueError(f"Checkpoint file not found: {path}")

    device = resolve_device("auto")
    state = torch.load(path, map_location=device, weights_only=False)
    config = ExperimentConfig.from_dict(state["config"])

    # Rebuild tokenizer
    tok_name = config.data.tokenizer_name
    if tok_name == "char":
        dataset = load_dataset(config.data.dataset_name)
        if isinstance(dataset, TextFileDataset):
            tokenizer = build_tokenizer(tok_name, text=dataset.text)
        else:
            tokenizer = build_tokenizer(tok_name, text="")
    else:
        tokenizer = build_tokenizer(tok_name)

    model = _build_model(config, tokenizer.vocab_size).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    return model, tokenizer, device


@app.get("/api/chat/status")
def get_chat_status():
    """Check if a model is loaded for chat."""
    with _chat_lock:
        return {
            "loaded": _chat_state["model"] is not None,
            "source": _chat_state["source"],
            "name": _chat_state["name"],
        }


@app.post("/api/chat/load")
async def load_chat_model(request: Request):
    """Load a model for chat generation."""
    body = json.loads(await request.body())
    source = body.get("source")

    with _chat_lock:
        # Free existing model
        _chat_state.update(model=None, tokenizer=None, source=None, name=None)

    if source == "hf":
        model_name = body.get("model_name")
        if model_name not in EVAL_MODELS:
            raise HTTPException(status_code=400, detail=f"Unknown model: {model_name}")

        from src.evaluation.hf_model import HFModel
        hf_id = EVAL_MODELS[model_name]
        hf_model = HFModel(hf_id)

        with _chat_lock:
            _chat_state.update(
                model=hf_model, tokenizer=hf_model.tokenizer,
                source="hf", name=model_name,
            )
        return {"status": "loaded", "name": model_name}

    elif source == "checkpoint":
        run_id = body.get("run_id")
        step = body.get("step")
        if run_id is None or step is None:
            raise HTTPException(status_code=400, detail="run_id and step required")

        try:
            model, tokenizer, device = _load_checkpoint_model(run_id, step)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))

        display_name = f"run-{run_id} step {step}"
        with _chat_lock:
            _chat_state.update(
                model=model, tokenizer=tokenizer,
                source="checkpoint", name=display_name,
            )
        return {"status": "loaded", "name": display_name}

    else:
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}")


@app.post("/api/chat/generate")
async def chat_generate(request: Request):
    """Generate text from the loaded chat model."""
    import torch

    with _chat_lock:
        model = _chat_state["model"]
        tokenizer = _chat_state["tokenizer"]
        source = _chat_state["source"]

    if model is None:
        raise HTTPException(status_code=400, detail="No model loaded. Call /api/chat/load first.")

    body = json.loads(await request.body())
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens", 128)
    temperature = body.get("temperature", 0.8)
    top_k = body.get("top_k", 50)

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    token_ids = tokenizer.encode(prompt)

    if source == "hf":
        # HFModel has its own generate method
        output_ids = model.generate(
            prompt_ids=token_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
        )
        text = tokenizer.decode(output_ids)
    else:
        # BaseModel.generate takes a tensor
        device = next(model.parameters()).device
        input_tensor = torch.tensor([token_ids], device=device)
        output_tensor = model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        # output includes prompt tokens — decode only new ones
        new_ids = output_tensor[0, len(token_ids):].tolist()
        text = tokenizer.decode(new_ids)

    return {"text": text, "prompt": prompt}


@app.post("/api/chat/unload")
def unload_chat_model():
    """Unload the chat model to free memory."""
    with _chat_lock:
        _chat_state.update(model=None, tokenizer=None, source=None, name=None)
    return {"status": "unloaded"}


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


