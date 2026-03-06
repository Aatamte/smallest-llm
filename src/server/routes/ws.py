"""WebSocket endpoint with sync protocol."""

from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.server.broadcast import broadcaster, _sanitize
from src.server.state import run_manager

router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    db = run_manager.db
    try:
        # 1. Send schemas
        await ws.send_text(json.dumps({
            "type": "schema",
            "tables": _sanitize(db.get_schemas()),
        }))

        # 2. Receive hashes from frontend
        raw = await ws.receive_text()
        msg = json.loads(raw)
        frontend_hashes = msg.get("hashes", {})

        # 3. Compare hashes, send dumps for mismatched tables
        backend_hashes = db.get_hashes()
        for name, backend_hash in backend_hashes.items():
            if frontend_hashes.get(name) != backend_hash:
                table = db.get_table(name)
                rows = table.get_all_rows()
                await ws.send_text(json.dumps(_sanitize({
                    "type": "dump",
                    "table": name,
                    "rows": rows,
                })))

        # 4. Send ready
        await ws.send_text(json.dumps({"type": "ready"}))

        # 5. Stream CDC ops
        async for message in broadcaster.subscribe():
            await ws.send_text(message)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
