"""Reply catalogue: every key the code can emit exists, every slot combination expands, names match the runtime."""
import ast
import json
import re
import sys
from pathlib import Path

import pytest

from runtime.actuators.info import CONDITIONS
from runtime.feedback import Feedback, clip_basename, render_text
from vcm import labels as L

ROOT = Path(__file__).resolve().parent.parent
REPLIES = json.loads((ROOT / "config" / "replies.json").read_text())
KEYS = {k: v for k, v in REPLIES.items() if not k.startswith("_")}
sys.path.insert(0, str(ROOT / "scripts"))
import gen_reply_audio as G  # noqa: E402


def _keys_of(node):
    """reply keys a `say=` expression can evaluate to: a literal, either branch of a conditional, or the first
    element of each (key, values) pair in a list"""
    if isinstance(node, ast.Constant):
        return {node.value} if isinstance(node.value, str) else set()
    if isinstance(node, ast.IfExp):
        return _keys_of(node.body) | _keys_of(node.orelse)
    if isinstance(node, ast.List):
        return set().union(*[_keys_of(e) for e in node.elts]) if node.elts else set()
    if isinstance(node, ast.Tuple) and node.elts:
        return _keys_of(node.elts[0])
    return set()


def keys_in_source():
    """every reply key literal used as say=... / say = ... / ERROR_REPLY value in runtime/"""
    found = set()
    for f in (ROOT / "runtime").rglob("*.py"):
        for n in ast.walk(ast.parse(f.read_text())):
            if isinstance(n, ast.keyword) and n.arg == "say":
                found |= _keys_of(n.value)
            if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "say" for t in n.targets):
                found |= _keys_of(n.value)
            if isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "ERROR_REPLY" for t in n.targets):
                found |= {v.value for v in n.value.values}
            if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "key" for t in n.targets)
                    and isinstance(n.value, ast.Subscript) and isinstance(n.value.value, ast.Dict)):
                found |= {v.value for v in n.value.value.values}      # clock.py ring keys
    return found


def test_every_reply_key_in_the_code_is_in_the_catalogue():
    used = keys_in_source()
    assert {"music_play", "oos_sorry", "timer_set", "weather", "time_now", "music_offline", "timer_ring", "alarm_ring"} <= used
    missing = used - set(KEYS)
    assert not missing, f"keys used in runtime/ but missing from config/replies.json: {sorted(missing)}"


def test_dispatcher_own_keys_are_catalogued():
    from runtime.dispatcher import DISPATCH_REPLY_KEYS
    assert DISPATCH_REPLY_KEYS <= set(KEYS)


def test_slot_values_in_catalogue_match_the_model_labels():
    assert KEYS["timer_set"]["slots"][0]["values"] == L.SLOT_VALUES["TIMER"]
    assert KEYS["alarm_set"]["slots"][0]["values"] == L.SLOT_VALUES["ALARM"]
    assert [f"{n} degrees" for n in KEYS["thermostat_set"]["slots"][0]["values"]] == L.SLOT_VALUES["TEMPERATURE"]
    assert [f"{n} percent" for n in KEYS["light_brightness"]["slots"][0]["values"]] == L.SLOT_VALUES["BRIGHTNESS"]
    assert [c.lower() for c in L.SLOT_VALUES["COLOR"]] == KEYS["light_color"]["slots"][0]["values"]
    assert [t.lower() for t in L.SLOT_VALUES["CREATE_REMINDER"]] == KEYS["reminder_saved"]["slots"][0]["values"]
    assert KEYS["weather"]["slots"][1]["values"] == CONDITIONS


def test_expansion_counts_and_unique_filenames():
    rows = G.expand(REPLIES)
    assert len(rows) == len({r["file"] for r in rows})
    n = {k: sum(1 for r in rows if r["key"] == k) for k in KEYS}
    assert n["time_now"] == 12 * 60 * 2 and n["weather"] == 31 * len(CONDITIONS) and n["timer_set"] == 3
    assert all("{" not in r["text"] for r in rows)


def test_runtime_finds_the_clip_the_generator_would_write(tmp_path):
    fb = Feedback(replies_dir=tmp_path, catalogue=ROOT / "config" / "replies.json", play=False)
    for r in G.expand(REPLIES)[:50] + [x for x in G.expand(REPLIES) if x["key"] in ("time_now", "timer_set")][:20]:
        (tmp_path / (r["file"] + ".mp3")).write_bytes(b"x")
        text, path = fb.resolve(r["key"], r["values"])
        assert path and Path(path).stem == r["file"] and text == r["text"]


def test_text_rendering():
    assert render_text(KEYS["timer_set"], {"amount": "1 minute"}) == "Timer set for 1 minute."
    assert render_text(KEYS["time_now"], {"h": 4, "mm": 5, "ampm": "PM"}) == "It's 4:05 PM."
    assert render_text(KEYS["reminder_list_header"], {"n": 1}) == "You have 1 reminder."
    assert clip_basename("light_color", KEYS["light_color"], {"color": "blue"}) == "light_color__color-blue"


def test_unknown_key_is_spoken_literally(tmp_path):
    fb = Feedback(replies_dir=tmp_path, catalogue=ROOT / "config" / "replies.json", play=False)
    assert fb.resolve("Hello there", {}) == ("Hello there", None)
