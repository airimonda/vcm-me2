"""TIMER, ALARM, CREATE_REMINDER, LIST_REMINDERS -- real logic, shown on the dashboard.

Timers and alarms run as asyncio tasks (they really fire), are mirrored into the shared State's
timers/alarms lists, and ring through the speaker: a repeated chime + spoken clip for up to 60 s until
STOP is said or the dashboard's Stop button sends {"type": "dismiss"} (both call dismiss()). Reminders are
a persisted list (no time: the model's reminders carry only a topic).
"""
from __future__ import annotations

import asyncio
import datetime
import re
import time

from runtime.actuators.base import Actuator
from runtime.actuators.info import now_local
from runtime.command import Response

RING_TIMEOUT_S = 60
RING_PERIOD_S = 4
MAX_REMINDERS_SPOKEN = 5
UNIT_SECONDS = {"second": 1, "minute": 60, "hour": 3600}


def parse_duration(slot: str) -> int | None:
    """'30 seconds' / '1 minute' -> seconds"""
    m = re.match(r"\s*(\d+)\s*(second|minute|hour)s?\b", str(slot or ""), re.I)
    return int(m.group(1)) * UNIT_SECONDS[m.group(2).lower()] if m else None


def parse_clock(slot: str):
    """'6:00 AM' -> (6, 0, 'AM')"""
    m = re.match(r"\s*(\d{1,2}):(\d{2})\s*([AP]M)\s*$", str(slot or ""), re.I)
    return (int(m.group(1)), int(m.group(2)), m.group(3).upper()) if m else None


def next_occurrence(hour12: int, minute: int, ampm: str, now: datetime.datetime) -> datetime.datetime:
    h24 = (hour12 % 12) + (12 if ampm.upper() == "PM" else 0)
    cand = now.replace(hour=h24, minute=minute, second=0, microsecond=0)
    return cand + datetime.timedelta(days=1) if cand <= now else cand


class ClockActuator(Actuator):
    name = "clock"
    kind = "real"

    def __init__(self, notify=None, speak=None, chime=None, timezone=None):
        self.notify = notify or (lambda: None)          # sync callable(): persist + bus push
        self.speak = speak                               # async callable(key, **values) or None
        self.chime = chime                               # sync callable(name) or None
        self.timezone = timezone
        self._tasks = {}                                 # id -> pending asyncio.Task
        self._ring_tasks = {}                            # (kind, id) -> ringing loop task
        self._ring_key = None

    async def handle(self, cmd, state):
        c = cmd.command
        if c == "TIMER":
            return self._timer(cmd, state)
        if c == "ALARM":
            return self._alarm(cmd, state)
        if c == "CREATE_REMINDER":
            return self._create_reminder(cmd, state)
        if c == "LIST_REMINDERS":
            return self._list_reminders(state)
        return Response(say=None, ok=False)

    def status(self):
        return {"name": self.name, "kind": self.kind, "pending": len(self._tasks), "ringing": self._ring_key}

    def _timer(self, cmd, state):
        seconds = parse_duration(cmd.slot)
        if not seconds:
            return Response(say="reprompt_unclear", ok=False)
        tid = state.next_id("timers")
        state.data["timers"].append({"id": tid, "label": cmd.slot, "ends_at": time.time() + seconds})
        self.notify()
        self._tasks[tid] = asyncio.ensure_future(self._fire_after(seconds, "timer", tid, cmd.slot, state))
        return Response(say="timer_set", data={"amount": cmd.slot})

    def _alarm(self, cmd, state):
        parts = parse_clock(cmd.slot)
        if not parts:
            return Response(say="reprompt_unclear", ok=False)
        h, mm, ampm = parts
        now = now_local(self.timezone)
        seconds = max(1.0, (next_occurrence(h, mm, ampm, now) - now).total_seconds())
        aid = state.next_id("alarms")
        state.data["alarms"].append({"id": aid, "time": f"{h}:{mm:02d}", "ampm": ampm.lower(), "ringing": False})
        self.notify()
        self._tasks[aid] = asyncio.ensure_future(self._fire_after(seconds, "alarm", aid, f"{h}:{mm:02d} {ampm}", state))
        return Response(say="alarm_set", data={"time": f"{h}:{mm:02d} {ampm}"})

    def _create_reminder(self, cmd, state):
        topic = str(cmd.slot or "").strip()
        if not topic:
            return Response(say="reprompt_unclear", ok=False)
        state.data["reminders"].append({"id": state.next_id("reminders"), "topic": topic, "created": time.time()})
        self.notify()
        return Response(say="reminder_saved", data={"topic": topic.lower()})

    def _list_reminders(self, state):
        items = state.data["reminders"][-MAX_REMINDERS_SPOKEN:]
        if not items:
            return Response(say="reminder_list_empty")
        say = [("reminder_list_header", {"n": len(items)})]
        say += [("reminder_item", {"topic": r["topic"].lower()}) for r in items]
        return Response(say=say)

    # -- ringing ---------------------------------------------------------------------
    async def _fire_after(self, seconds, kind, item_id, label, state):
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            return
        self._tasks.pop(item_id, None)
        if kind == "alarm":
            for a in state.data["alarms"]:
                if a["id"] == item_id:
                    a["ringing"] = True
        task = asyncio.ensure_future(self._ring(kind, item_id, label, state))
        self._ring_tasks[(kind, item_id)] = task
        await task

    async def _ring(self, kind, item_id, label, state):
        state.data["ringing"] = {"kind": kind, "id": item_id, "label": label}
        self._ring_key = (kind, item_id)
        self.notify()
        key = {"timer": "timer_ring", "alarm": "alarm_ring"}[kind]
        start = time.monotonic()
        try:
            while time.monotonic() - start < RING_TIMEOUT_S:
                if self.chime:
                    self.chime("ring")
                if self.speak:
                    await self.speak(key)
                await asyncio.sleep(RING_PERIOD_S)
        except asyncio.CancelledError:
            pass
        finally:
            self._clear(kind, item_id, state)

    def is_ringing(self):
        return self._ring_key is not None

    def dismiss(self, state):
        """STOP or the dashboard's Stop button: silence whatever is ringing"""
        if not self._ring_key:
            return False
        task = self._ring_tasks.get(self._ring_key)
        if task and not task.done():
            task.cancel()
        else:
            self._clear(self._ring_key[0], self._ring_key[1], state)
        return True

    def _clear(self, kind, item_id, state):
        if kind == "timer":
            state.data["timers"] = [t for t in state.data["timers"] if t["id"] != item_id]
        else:
            state.data["alarms"] = [a for a in state.data["alarms"] if a["id"] != item_id]
        self._ring_tasks.pop((kind, item_id), None)
        if self._ring_key == (kind, item_id):
            self._ring_key = None
            state.data["ringing"] = None
        self.notify()

    def cancel(self):
        for t in list(self._tasks.values()) + list(self._ring_tasks.values()):
            t.cancel()
        self._tasks.clear()
        self._ring_tasks.clear()
