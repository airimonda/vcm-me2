"""Emulated thermostat / aircon (TEMPERATURE). Setpoint within [min, max] C and a simulated room
temperature that drifts 1 C every `drift_period_s` toward the setpoint while on (a real asyncio task, so
the dashboard sees it move during a run)."""
from __future__ import annotations

import asyncio
import re

from runtime.actuators.base import Actuator
from runtime.command import Response


class ThermostatActuator(Actuator):
    name = "thermostat"
    kind = "emulated"

    def __init__(self, notify=None, min_c=16, max_c=30, drift_period_s=120, start_room_temp=28.0):
        self.notify = notify or (lambda: None)
        self.min_c, self.max_c = min_c, max_c
        self.drift_period_s = drift_period_s
        self.start_room_temp = start_room_temp
        self._drift_task = None
        self._init = False

    async def handle(self, cmd, state):
        ac = state.data["thermostat"]
        if not self._init:
            ac["room_temp"] = self.start_room_temp if not ac.get("on") else ac.get("room_temp", self.start_room_temp)
            self._init = True
        m = re.search(r"\d+", str(cmd.slot or ""))
        if cmd.command != "TEMPERATURE" or not m:
            return Response(say="reprompt_unclear", ok=False)
        n = int(m.group())
        if not (self.min_c <= n <= self.max_c):
            return Response(say="thermostat_out_of_range", ok=False)
        ac["setpoint"], ac["on"] = n, True
        self._ensure_drift(state)
        self.notify()
        return Response(say="thermostat_set", data={"n": n})

    def _ensure_drift(self, state):
        if self._drift_task is None or self._drift_task.done():
            self._drift_task = asyncio.ensure_future(self._drift(state))

    async def _drift(self, state):
        try:
            while True:
                await asyncio.sleep(self.drift_period_s)
                ac = state.data["thermostat"]
                if not ac["on"]:
                    return
                if abs(ac["room_temp"] - ac["setpoint"]) < 1.0:
                    ac["room_temp"] = float(ac["setpoint"])
                else:
                    ac["room_temp"] += -1.0 if ac["room_temp"] > ac["setpoint"] else 1.0
                self.notify()
        except asyncio.CancelledError:
            return

    def cancel(self):
        if self._drift_task:
            self._drift_task.cancel()
