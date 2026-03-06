"""FastAPI backend for the training dashboard."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.server.broadcast import _sanitize
from src.server.routes import chat, evals, models, presets, runs, ws
from src.server.state import DB_PATH, MODELS_DIR, download_preset_models, eval_db, run_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    from src.server.broadcast import broadcaster
    broadcaster.set_loop(asyncio.get_running_loop())
    run_manager.recover_stale()
    download_preset_models()
    run_manager.db.sync_models_dir(MODELS_DIR)
    yield
    run_manager.shutdown()
    eval_db.close()


class SafeJSONResponse(JSONResponse):
    """JSONResponse that replaces NaN/Inf with null."""
    def render(self, content) -> bytes:
        return json.dumps(_sanitize(content)).encode("utf-8")


app = FastAPI(title="smallest-llm dashboard", lifespan=lifespan, default_response_class=SafeJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(runs.router)
app.include_router(evals.router)
app.include_router(models.router)
app.include_router(presets.router)
app.include_router(chat.router)
app.include_router(ws.router)
