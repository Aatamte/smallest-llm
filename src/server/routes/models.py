"""Model management endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from src.server.state import run_manager

router = APIRouter()


@router.get("/api/models")
def list_models():
    return run_manager.db.list_models()


@router.delete("/api/models/{model_id}")
def delete_model(model_id: int):
    """Delete a saved model (DB record + files on disk)."""
    import os
    import shutil

    model = run_manager.db.get_model(model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")

    path = model.get("path", "")
    if path and os.path.isdir(path):
        shutil.rmtree(path)

    run_manager.db.delete_model(model_id)
    return {"status": "deleted", "model_id": model_id}
