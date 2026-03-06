"""Chat / generation endpoints."""

from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request

from src.server.state import chat_service

router = APIRouter()


@router.get("/api/chat/status")
def get_chat_status():
    return chat_service.get_status()


@router.post("/api/chat/load")
async def load_chat_model(request: Request):
    body = json.loads(await request.body())
    source = body.get("source")

    try:
        return chat_service.load(
            source,
            model_name=body.get("model_name"),
            run_id=body.get("run_id"),
            step=body.get("step"),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/chat/generate")
async def chat_generate(request: Request):
    body = json.loads(await request.body())
    prompt = body.get("prompt", "")

    if not prompt:
        raise HTTPException(status_code=400, detail="prompt is required")

    try:
        return chat_service.generate(
            prompt=prompt,
            max_tokens=body.get("max_tokens", 128),
            temperature=body.get("temperature", 0.8),
            top_k=body.get("top_k", 50),
        )
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/api/chat/unload")
def unload_chat_model():
    return chat_service.unload()
