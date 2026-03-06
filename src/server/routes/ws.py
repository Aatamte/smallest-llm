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
        schemas = db.get_schemas()
        print(f"[WS] Sending schema: {[s.get('name', s) for s in schemas]}")
        await ws.send_text(json.dumps({
            "type": "schema",
            "tables": _sanitize(schemas),
        }))

        # 2. Receive hashes from frontend
        raw = await ws.receive_text()
        msg = json.loads(raw)
        frontend_hashes = msg.get("hashes", {})
        print(f"[WS] Frontend hashes: {frontend_hashes}")

        # 3. Compare hashes, send dumps for mismatched tables
        backend_hashes = db.get_hashes()
        print(f"[WS] Backend hashes: {backend_hashes}")
        CHUNK_SIZE = 5000
        for name, backend_hash in backend_hashes.items():
            if frontend_hashes.get(name) != backend_hash:
                table = db.get_table(name)
                rows = table.get_all_rows()
                print(f"[WS] Dumping table '{name}': {len(rows)} rows (frontend={frontend_hashes.get(name)}, backend={backend_hash})")
                for i in range(0, len(rows), CHUNK_SIZE):
                    chunk = rows[i:i + CHUNK_SIZE]
                    payload = json.dumps(_sanitize({"type": "dump", "table": name, "rows": chunk}))
                    print(f"[WS]   chunk {i//CHUNK_SIZE + 1}: {len(chunk)} rows, {len(payload)} bytes")
                    await ws.send_text(payload)
            else:
                print(f"[WS] Skipping table '{name}': hashes match ({backend_hash})")

        # 4. Send ready
        print(f"[WS] Sending ready")
        await ws.send_text(json.dumps({"type": "ready"}))

        # 5. Stream CDC ops
        async for message in broadcaster.subscribe():
            await ws.send_text(message)
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
