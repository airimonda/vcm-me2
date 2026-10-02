"""In-process asyncio pub/sub between the runtime and the dashboard websocket clients.

server -> client: {"type": "state"|"heard"|"reply"|"turn"|"answer"|"metrics"|"log", ...}
client -> server: {"type": "command"|"dismiss"|"mark", ...} on `inbound`.
"""
from __future__ import annotations

import asyncio


class Bus:
    def __init__(self, maxsize=256):
        self._subscribers = []
        self._maxsize = maxsize
        self.inbound = asyncio.Queue()

    def subscribe(self):
        q = asyncio.Queue(maxsize=self._maxsize)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        if q in self._subscribers:
            self._subscribers.remove(q)

    def publish(self, message):
        """non-blocking fan-out; a full queue (stalled client) drops its oldest message"""
        for q in list(self._subscribers):
            try:
                q.put_nowait(message)
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    q.put_nowait(message)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass

    async def send(self, message):
        await self.inbound.put(message)

    def send_nowait(self, message):
        self.inbound.put_nowait(message)
