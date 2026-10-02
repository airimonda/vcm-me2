"""One JSON-serialisable state object, shaped like the dashboard's {"type":"state","state":{...}} payload.
Actuators mutate their own top-level key; the runtime calls save() and publishes a snapshot on the bus."""
from __future__ import annotations

import json
import os
import time

DEFAULT_STATE = {
    "mode": "idle",                 # idle | listening | thinking | speaking
    "music": {"connected": False, "playing": False, "paused": False, "track": None, "volume": None,
              "device": None, "provider": "spotify", "error": None},
    "light": {"connected": True, "on": False, "brightness": 60, "color": "#FFFFFF", "color_name": "White"},
    "thermostat": {"on": False, "setpoint": 24, "room_temp": 28.0},
    "timers": [],                   # [{id, label, ends_at}]
    "alarms": [],                   # [{id, time, ampm, ringing}]
    "reminders": [],                # [{id, topic, created}]
    "ringing": None,                # null | {kind, id, label}
    "call": None,                   # null | {action, contact, started_at}
    "weather": None,                # last weather answer {temp, condition, city, t}
}


def _copy(d):
    return json.loads(json.dumps(d))


class State:
    def __init__(self, path=None):
        self.path = str(path) if path else None
        self.data = _copy(DEFAULT_STATE)
        self._seq = {}
        if self.path and os.path.exists(self.path):
            try:
                loaded = json.load(open(self.path))
                for k, v in loaded.items():
                    if k in self.data and isinstance(self.data[k], dict) and isinstance(v, dict):
                        self.data[k].update(v)
                    elif k in self.data:
                        self.data[k] = v
                # timers/alarms are asyncio tasks that do not survive a restart; keep only the reminders
                self.data["timers"], self.data["alarms"], self.data["ringing"], self.data["call"] = [], [], None, None
                self.data["mode"] = "idle"
            except (ValueError, OSError):
                pass
        for kind, prefix in (("reminders", "r"),):
            ids = [int(x["id"][1:]) for x in self.data[kind] if str(x.get("id", "")).startswith(prefix)]
            self._seq[kind] = max(ids, default=0)

    def save(self):
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        os.replace(tmp, self.path)

    def snapshot(self):
        return _copy(self.data)

    def set_mode(self, mode):
        self.data["mode"] = mode

    def next_id(self, kind):
        prefix = {"timers": "t", "alarms": "a", "reminders": "r"}[kind]
        self._seq[kind] = self._seq.get(kind, 0) + 1
        return f"{prefix}{self._seq[kind]}"
