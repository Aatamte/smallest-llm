"""Evaluation endpoints — run evals, check status, list results."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from src.server.state import eval_db, eval_service, run_manager
from src.services.eval_service import HARNESS_ALL_TASKS, HARNESS_PRESETS
from src.types.eval import EvalRecord

router = APIRouter()


@router.get("/api/evals")
def list_all_evals(model_name: Optional[str] = Query(None), task: Optional[str] = Query(None)):
    rows = eval_db.get_evals(model_name=model_name, task=task)
    return [EvalRecord.from_db_row(r).to_wire() for r in rows]


@router.get("/api/evals/models")
def list_eval_models():
    return eval_db.list_models()


@router.get("/api/evals/available-models")
def list_available_models():
    models = run_manager.db.list_models()
    return [{"name": m["name"], "path": m["path"]} for m in models]


@router.get("/api/evals/available-harness-tasks")
def list_available_harness_tasks():
    return {"presets": HARNESS_PRESETS, "tasks": HARNESS_ALL_TASKS}


@router.get("/api/evals/status")
def get_eval_status():
    return eval_service.get_status()


@router.post("/api/evals/stop")
def stop_eval():
    return eval_service.stop()


@router.get("/api/eval-presets")
def list_eval_presets():
    from src.evaluation.config import get_eval_presets
    return get_eval_presets()


@router.get("/api/eval-presets/{name}")
def get_eval_preset_config(name: str):
    from src.evaluation.config import get_eval_preset
    preset = get_eval_preset(name)
    if preset is None:
        raise HTTPException(status_code=404, detail=f"Eval preset '{name}' not found")
    return {
        "eval_tasks": len(preset.tasks) > 0,
        "eval_tasks_list": ",".join(preset.tasks),
        "eval_tasks_interval": preset.interval,
        "eval_tasks_max_samples": preset.max_samples or 0,
    }


@router.post("/api/evals/run")
async def run_eval(request: Request):
    body = json.loads(await request.body())
    source = body.get("source", "hf")
    tasks = body.get("tasks", [])
    harness_tasks = body.get("harness_tasks", [])

    if not tasks and not harness_tasks:
        raise HTTPException(status_code=400, detail="No tasks specified")

    # Validate native tasks
    from src.evaluation.tasks import list_tasks
    available_tasks = list_tasks()
    invalid = [t for t in tasks if t not in available_tasks]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown tasks: {invalid}. Available: {available_tasks}")

    # Validate harness tasks
    invalid_harness = [t for t in harness_tasks if t not in HARNESS_ALL_TASKS]
    if invalid_harness:
        raise HTTPException(status_code=400, detail=f"Unknown harness tasks: {invalid_harness}. Available: {HARNESS_ALL_TASKS}")

    try:
        result = eval_service.start(
            source=source,
            tasks=tasks,
            harness_tasks=harness_tasks,
            max_samples=body.get("max_samples"),
            harness_limit=body.get("harness_limit"),
            model_name=body.get("model_name"),
            run_id=body.get("run_id"),
            step=body.get("step"),
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=409, detail=str(e))
