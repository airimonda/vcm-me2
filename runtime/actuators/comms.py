"""Emulated phone (CALL, MESSAGE): a "Calling X" / "Message to X" card on the dashboard for a few seconds.
The model has no contact slot, so every call/message goes to `comms.contact`; the message text is fixed."""
from __future__ import annotations

import asyncio
import time

from runtime.actuators.base import Actuator
from runtime.command import Response

CARD_DURATION_S = 3.0


class CommsActuator(Actuator):
    name = "comms"
    kind = "emulated"

    def __init__(self, notify=None, contact="Mom", message_text="I'm on my way", card_duration_s=CARD_DURATION_S):
        self.notify = notify or (lambda: None)
        self.contact = contact
        self.message_text = message_text
        self.card_duration_s = card_duration_s
        self._clear_task = None

    async def handle(self, cmd, state):
        action = "message" if cmd.command == "MESSAGE" else "call"
        state.data["call"] = {"action": action, "contact": self.contact, "text": self.message_text if action == "message" else None,
                              "started_at": time.time()}
        self.notify()
        if self._clear_task and not self._clear_task.done():
            self._clear_task.cancel()
        self._clear_task = asyncio.ensure_future(self._clear_after(state))
        return Response(say="comms_message" if action == "message" else "comms_call", data={"contact": self.contact})

    async def _clear_after(self, state):
        try:
            await asyncio.sleep(self.card_duration_s)
        except asyncio.CancelledError:
            return
        state.data["call"] = None
        self.notify()

    def cancel(self):
        if self._clear_task:
            self._clear_task.cancel()
