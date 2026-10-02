"""Value objects passed between the model, the dispatcher and the actuators. Plain data, no behaviour."""
from __future__ import annotations

import time


class Command:
    """A decided command: one of the 19 commands (+ slot value string) or OUT_OF_SCOPE.

    source: "voice" (mic / wav through the model) | "client" (dashboard test button)."""

    def __init__(self, command, slot=None, prob=1.0, slot_prob=None, raw_command=None, t=None, source="voice"):
        self.command = command
        self.slot = slot
        self.prob = float(prob)
        self.slot_prob = slot_prob
        self.raw_command = raw_command or command
        self.t = t if t is not None else time.time()
        self.source = source

    @classmethod
    def from_decision(cls, d, source="voice"):
        return cls(d.command, d.slot, d.prob, d.slot_prob, d.raw_command, source=source)

    def to_dict(self):
        return {"command": self.command, "slot": self.slot, "prob": self.prob, "slot_prob": self.slot_prob,
                "raw_command": self.raw_command, "t": self.t, "source": self.source}

    def __repr__(self):
        s = f", slot={self.slot!r}" if self.slot else ""
        return f"Command({self.command}{s}, p={self.prob:.2f}, source={self.source})"


class Response:
    """What an actuator's handle() (or the dispatcher itself) produced.

    say: a reply key in config/replies.json, a list of (key, values) pairs spoken in sequence, or None
         (silent turn).
    sound: earcon name played before the reply ("success" | "reject"), or None.
    ok: the action succeeded (a failed response still usually has something to say).
    data: template values for `say` when it is a single key; may also carry the displayed answer.
    accepted / reason: set by the dispatcher (accepted = the command was acted on).
    """

    def __init__(self, say=None, sound=None, ok=True, data=None, accepted=True, reason="", answer=None):
        self.say = say
        self.sound = sound
        self.ok = bool(ok)
        self.data = dict(data or {})
        self.accepted = bool(accepted)
        self.reason = reason
        self.answer = answer          # optional dict for the dashboard "Last answer" (weather/time)

    def __repr__(self):
        return f"Response(say={self.say!r}, ok={self.ok}, accepted={self.accepted}, reason={self.reason!r})"
