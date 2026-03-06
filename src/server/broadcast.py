"""In-process async broadcast channel for live WebSocket streaming."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any


def _sanitize(obj):
    """Replace NaN/Inf with None so JSON serialization doesn't fail."""
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


class Broadcaster:
    """Publish/subscribe broadcast for live training metrics.

    The trainer (sync) calls publish() to push metrics.
    WebSocket handlers (async) iterate over subscribe() to receive them.
    """

    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None
        self.text_state: str = ""

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop (called once at server startup)."""
        self._loop = loop

    def publish(self, message: dict[str, Any]):
        """Publish a message to all subscribers. Thread-safe."""
        if message.get("type") == "textState":
            self.text_state = message.get("data", "")
        data = json.dumps(_sanitize(message))
        self._send(data)

    def publish_op(
        self,
        table: str,
        op: str,
        row: dict | None = None,
        key: Any = None,
    ):
        """Publish a single CDC operation to all subscribers. Thread-safe."""
        msg: dict[str, Any] = {"type": "op", "table": table, "op": op}
        if row is not None:
            msg["row"] = row
        if key is not None:
            msg["key"] = key
        data = json.dumps(_sanitize(msg))
        self._send(data)

    def publish_ops(
        self,
        table: str,
        op: str,
        rows: list[dict],
    ):
        """Publish a batch of CDC operations as one message. Thread-safe."""
        if not rows:
            return
        msg: dict[str, Any] = {"type": "ops", "table": table, "op": op, "rows": rows}
        data = json.dumps(_sanitize(msg))
        self._send(data)

    def _send(self, data: str):
        """Push serialized JSON to all subscriber queues."""
        for queue in list(self._subscribers):
            if self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(queue.put_nowait, data)
            else:
                try:
                    queue.put_nowait(data)
                except asyncio.QueueFull:
                    pass

    async def subscribe(self):
        """Async generator yielding messages as they arrive."""
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        self._subscribers.append(queue)
        try:
            while True:
                data = await queue.get()
                yield data
        finally:
            self._subscribers.remove(queue)


# Global broadcaster instance — shared between trainer and server
broadcaster = Broadcaster()
