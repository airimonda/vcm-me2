"""Emulated colour light (LIGHT_ON, LIGHT_OFF, BRIGHTNESS, COLOR). No hardware: the state is shown on the
dashboard tile. The old BLE bulb driver was dropped (the demo bulb speaks an encrypted Tuya protocol), so
`mock.light` in config/runtime.yaml is always effectively true."""
from __future__ import annotations

import re

from runtime.actuators.base import Actuator
from runtime.command import Response

COLOR_TABLE = {"red": "#FF3B30", "blue": "#0A84FF", "green": "#34C759"}


def parse_percent(slot) -> int | None:
    m = re.search(r"\d+", str(slot or ""))
    return int(m.group()) if m else None


class LightActuator(Actuator):
    name = "light"
    kind = "emulated"

    def __init__(self, notify=None):
        self.notify = notify or (lambda: None)

    async def handle(self, cmd, state):
        li = state.data["light"]
        li["connected"] = True
        c = cmd.command
        if c in ("LIGHT_ON", "LIGHT_OFF"):
            li["on"] = c == "LIGHT_ON"
            self.notify()
            return Response(say="lights_on" if li["on"] else "lights_off")
        if c == "BRIGHTNESS":
            n = parse_percent(cmd.slot)
            if n is None:
                return Response(say="reprompt_unclear", ok=False)
            li["brightness"] = max(1, min(100, n))
            li["on"] = True
            self.notify()
            return Response(say="light_brightness", data={"n": li["brightness"]})
        if c == "COLOR":
            name = str(cmd.slot or "").strip()
            if name.lower() not in COLOR_TABLE:
                return Response(say="reprompt_unclear", ok=False)
            li["color"], li["color_name"], li["on"] = COLOR_TABLE[name.lower()], name.capitalize(), True
            self.notify()
            return Response(say="light_color", data={"color": name.lower()})
        return Response(say=None, ok=False)
