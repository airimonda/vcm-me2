"""Dispatcher: every one of the 19 commands (+ slots) and OUT_OF_SCOPE reaches the right actuator and reply."""
import asyncio
import json
from pathlib import Path

import pytest

from runtime import dispatcher
from runtime.command import Command
from runtime.config import load_config
from runtime.devices import build_actuators
from runtime.state import State
from vcm import labels as L

ROOT = Path(__file__).resolve().parent.parent
REPLIES = json.loads((ROOT / "config" / "replies.json").read_text())
CFG = load_config(None, {"mock": {"spotify": True}, "thermostat": {"drift_period_s": 3600}})


def run(cmds, cfg=CFG, oos="sorry", setup=None):
    """dispatch a list of (command, slot) in one event loop; returns (responses, state, actuators)"""
    async def go():
        state = State()
        acts = build_actuators(cfg, notify=lambda: None)
        acts["info"]._fetch = lambda: (31, "partly cloudy")
        if setup:
            setup(acts)
        out = []
        try:
            for c, slot in cmds:
                out.append(await dispatcher.dispatch(Command(c, slot), state, acts, oos))
        finally:
            for a in acts.values():
                a.cancel()
        return out, state, acts
    return asyncio.run(go())


ALL = [(c, None) for c in L.COMMANDS if c not in L.SLOT_VALUES] + \
      [(c, v) for c, vals in L.SLOT_VALUES.items() for v in vals]


def test_the_mapping_covers_exactly_the_19_commands():
    assert set(dispatcher.ACTUATOR_FOR) == set(L.COMMANDS) and len(L.COMMANDS) == 19
    assert set(dispatcher.ACTUATOR_FOR.values()) <= {"music", "light", "thermostat", "comms", "clock", "info"}


@pytest.mark.parametrize("command,slot", ALL)
def test_every_command_is_accepted_and_has_a_catalogued_reply(command, slot):
    # NEXT needs music playing in the mock player
    cmds = [("PLAY_MUSIC", None), (command, slot)] if command == "NEXT" else [(command, slot)]
    resps, _, _ = run(cmds)
    r = resps[-1]
    assert r.accepted and r.ok, (command, slot, r)
    keys = [k for k, _ in r.say] if isinstance(r.say, list) else [r.say]
    assert keys and all(k in REPLIES for k in keys), keys


def test_music_commands_drive_the_music_actuator():
    resps, state, acts = run([("PLAY_MUSIC", None), ("VOLUME_UP", None), ("VOLUME_DOWN", None), ("NEXT", None),
                              ("PAUSE", None), ("STOP", None)])
    assert [r.say for r in resps] == ["music_play", "music_volume_up", "music_volume_down", "music_next",
                                      "music_paused", "music_stopped"]
    assert acts["music"].player.calls == ["play", "volume+15", "volume-15", "next", "pause", "pause"]
    assert state.data["music"]["stopped"] is True


def test_light_state_brightness_color():
    resps, state, _ = run([("LIGHT_ON", None), ("BRIGHTNESS", "20 percent"), ("COLOR", "Green"), ("LIGHT_OFF", None)])
    assert [r.say for r in resps] == ["lights_on", "light_brightness", "light_color", "lights_off"]
    assert resps[1].data == {"n": 20} and resps[2].data == {"color": "green"}
    li = state.data["light"]
    assert li["brightness"] == 20 and li["color_name"] == "Green" and li["color"] == "#34C759" and li["on"] is False


def test_brightness_and_color_turn_the_light_on():
    _, state, _ = run([("BRIGHTNESS", "100 percent")])
    assert state.data["light"]["on"] and state.data["light"]["brightness"] == 100
    _, state, _ = run([("COLOR", "Red")])
    assert state.data["light"]["on"] and state.data["light"]["color_name"] == "Red"


def test_temperature_sets_the_thermostat():
    resps, state, _ = run([("TEMPERATURE", "26 degrees")])
    assert resps[0].say == "thermostat_set" and resps[0].data == {"n": 26}
    assert state.data["thermostat"]["setpoint"] == 26 and state.data["thermostat"]["on"]


def test_timer_alarm_and_reminders():
    resps, state, _ = run([("TIMER", "30 seconds"), ("ALARM", "9:00 PM"), ("CREATE_REMINDER", "Drink water"),
                           ("CREATE_REMINDER", "Study"), ("LIST_REMINDERS", None)])
    assert resps[0].say == "timer_set" and resps[0].data == {"amount": "30 seconds"}
    assert state.data["timers"][0]["label"] == "30 seconds"
    assert state.data["timers"][0]["ends_at"] > 0
    assert resps[1].say == "alarm_set" and resps[1].data == {"time": "9:00 PM"}
    assert state.data["alarms"][0]["time"] == "9:00" and state.data["alarms"][0]["ampm"] == "pm"
    assert [r["topic"] for r in state.data["reminders"]] == ["Drink water", "Study"]
    assert resps[3].say == "reminder_saved" and resps[3].data == {"topic": "study"}
    assert resps[4].say == [("reminder_list_header", {"n": 2}), ("reminder_item", {"topic": "drink water"}),
                            ("reminder_item", {"topic": "study"})]


def test_list_reminders_empty():
    assert run([("LIST_REMINDERS", None)])[0][0].say == "reminder_list_empty"


def test_comms_and_info():
    resps, state, _ = run([("CALL", None), ("MESSAGE", None), ("WEATHER", None), ("TIME", None)])
    assert resps[0].say == "comms_call" and resps[0].data == {"contact": "Mom"}
    assert resps[1].say == "comms_message"
    assert state.data["call"]["action"] == "message" and state.data["call"]["text"] == "I'm on my way"
    assert resps[2].say == "weather" and resps[2].data["temp"] == 31 and resps[2].data["condition"] == "partly cloudy"
    assert state.data["weather"]["temp"] == 31 and resps[2].answer["kind"] == "weather"
    assert resps[3].say == "time_now" and set(resps[3].data) == {"h", "mm", "ampm"} and resps[3].data["ampm"] in ("AM", "PM")


def test_weather_offline_reply():
    def boom():
        raise OSError("no route")
    resps, _, _ = run([("WEATHER", None)], setup=lambda a: setattr(a["info"], "_fetch", boom))
    assert resps[0].say == "weather_offline" and resps[0].ok is False


# ---------------------------------------------------------------- OUT_OF_SCOPE
def test_out_of_scope_says_sorry_by_default():
    r = run([("OUT_OF_SCOPE", None)])[0][0]
    assert r.say == "oos_sorry" and r.accepted is False and r.reason == "out_of_scope"
    assert REPLIES["oos_sorry"]["text"] == "Sorry, I didn't catch that."


def test_out_of_scope_silent_mode():
    r = run([("OUT_OF_SCOPE", None)], oos="silent")[0][0]
    assert r.say is None and r.accepted is False and r.sound == "reject"


def test_oos_touches_no_device():
    resps, state, acts = run([("OUT_OF_SCOPE", None)])
    assert acts["music"].player.calls == [] and state.data["timers"] == [] and state.data["light"]["on"] is False


def test_unknown_command_is_rejected_not_crashed():
    r = run([("NO_SPEECH", None)])[0][0]
    assert r.accepted is False and r.reason == "unknown_command"


# ---------------------------------------------------------------- ringing + STOP
def test_stop_dismisses_a_ringing_timer_instead_of_the_music():
    async def go():
        state = State()
        acts = build_actuators(CFG, notify=lambda: None, speak=None, chime=None)
        clock = acts["clock"]
        clock_start = await dispatcher.dispatch(Command("TIMER", "10 seconds"), state, acts)
        task_id = state.data["timers"][0]["id"]
        clock._tasks[task_id].cancel()                    # fast-forward: ring now instead of waiting 10 s
        ring = asyncio.ensure_future(clock._ring("timer", task_id, "10 seconds", state))
        clock._ring_tasks[("timer", task_id)] = ring
        await asyncio.sleep(0.05)
        assert state.data["ringing"]["kind"] == "timer" and clock.is_ringing()
        r = await dispatcher.dispatch(Command("STOP"), state, acts)
        await asyncio.sleep(0.05)
        out = (clock_start, r, state.data["ringing"], state.data["timers"], acts["music"].player.calls)
        for a in acts.values():
            a.cancel()
        return out
    start, r, ringing, timers, music_calls = asyncio.run(go())
    assert r.reason == "dismissed_ring" and ringing is None and timers == [] and music_calls == []


def test_timer_really_fires(monkeypatch):
    from runtime.actuators import clock as clockmod
    monkeypatch.setattr(clockmod, "RING_PERIOD_S", 0.05)
    spoken, chimes = [], []

    async def speak(key, **v):
        spoken.append(key)

    async def go():
        state = State()
        acts = build_actuators(CFG, notify=lambda: None, speak=speak, chime=chimes.append)
        monkeypatch.setattr(clockmod, "parse_duration", lambda s: 0.1)       # 0.1 s timer
        await dispatcher.dispatch(Command("TIMER", "10 seconds"), state, acts)
        await asyncio.sleep(0.4)
        was_ringing = acts["clock"].is_ringing()
        acts["clock"].dismiss(state)
        await asyncio.sleep(0.05)
        for a in acts.values():
            a.cancel()
        return was_ringing, state.data["ringing"]
    was_ringing, after = asyncio.run(go())
    assert was_ringing and after is None and "timer_ring" in spoken and "ring" in chimes
