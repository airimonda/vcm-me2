"""Routing: decided command (+ slot) -> actuator action -> Response (reply key, sound, data).

dispatch() is the single entry point the runtime and tests call. It knows nothing about audio, the bus or
playback; it returns a runtime.command.Response and the caller speaks it.

Confidence is handled upstream by the model's decision rule (max prob < tau -> OUT_OF_SCOPE), so every
command that arrives here is acted on.
"""
from __future__ import annotations

from runtime.command import Response

# command -> actuator key (see runtime.app.build_actuators)
ACTUATOR_FOR = {
    "PLAY_MUSIC": "music", "PAUSE": "music", "STOP": "music", "NEXT": "music",
    "VOLUME_UP": "music", "VOLUME_DOWN": "music",
    "LIGHT_ON": "light", "LIGHT_OFF": "light", "BRIGHTNESS": "light", "COLOR": "light",
    "TEMPERATURE": "thermostat",
    "CALL": "comms", "MESSAGE": "comms",
    "TIMER": "clock", "ALARM": "clock", "CREATE_REMINDER": "clock", "LIST_REMINDERS": "clock",
    "WEATHER": "info", "TIME": "info",
}

# every reply key the dispatcher itself can produce (the actuators' keys are checked against
# config/replies.json by tests/test_runtime_replies.py)
DISPATCH_REPLY_KEYS = {"oos_sorry"}


async def dispatch(cmd, state, actuators, oos_reply="sorry"):
    """cmd: runtime.command.Command; state: runtime.state.State; actuators: dict keyed like ACTUATOR_FOR's values.
    oos_reply: "sorry" (say "Sorry, I didn't catch that.") or "silent"."""
    if cmd.command == "OUT_OF_SCOPE":
        say = "oos_sorry" if oos_reply == "sorry" else None
        return Response(say=say, sound="reject", ok=True, accepted=False, reason="out_of_scope")
    key = ACTUATOR_FOR.get(cmd.command)
    if key is None:
        return Response(say=None, ok=False, accepted=False, reason="unknown_command")

    # STOP while a timer/alarm rings silences the ring instead of touching the music
    clock = actuators.get("clock")
    if cmd.command == "STOP" and clock is not None and clock.is_ringing():
        clock.dismiss(state)
        return Response(say=None, ok=True, accepted=True, reason="dismissed_ring")

    actuator = actuators.get(key)
    if actuator is None:
        return Response(say=None, ok=False, accepted=False, reason="no_actuator")
    resp = await actuator.handle(cmd, state)
    resp.accepted = True
    return resp
