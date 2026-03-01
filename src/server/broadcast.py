"""In-process async broadcast channel for live WebSocket streaming."""

from __future__ import annotations

import asyncio
import json
from typing import Any


class Broadcaster:
    """Publish/subscribe broadcast for live training metrics.

    The trainer (sync) calls publish() to push metrics.
    WebSocket handlers (async) iterate over subscribe() to receive them.
    """

    def __init__(self):
        self._subscribers: list[asyncio.Queue] = []
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop (called once at server startup)."""
        self._loop = loop

    def publish(self, message: dict[str, Any]):
        """Publish a message to all subscribers. Thread-safe."""
        data = json.dumps(message)
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
