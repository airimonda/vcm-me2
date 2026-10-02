"""Config, metrics, conversation text, dashboard websocket, Spotify auth helper."""
import asyncio
import json
import sys
from pathlib import Path

import pytest

from runtime.config import DEFAULTS, load_config, parse_mock
from runtime.convo import ConvoLog, rebuild_text
from runtime.command import Command
from runtime.metrics import Metrics, Turn
from vcm import labels as L

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- config
def test_default_config_file_matches_defaults_and_points_at_the_released_models():
    cfg = load_config(None)
    assert cfg == load_config(None, {}) and (ROOT / cfg["model"]["path"]).exists()
    assert (ROOT / cfg["model"]["path"]).with_suffix(".json").exists()
    assert cfg["model"]["path"] == "models/vcm_conformer_M_ens3.onnx" and cfg["model"]["tau"] is None
    assert cfg["wake"]["path"] == "models/wake/vcm_wake_int8.onnx" and cfg["wake"]["threshold"] == 0.55
    assert cfg["wake"]["hop_s"] == 0.25 and (ROOT / cfg["wake"]["path"]).exists()
    assert cfg["audio"]["earcon_volume"] == 0.15
    assert cfg["music"]["device_name"] == "Watson"
    assert set(cfg) == set(DEFAULTS)


def test_overrides_and_missing_file(tmp_path):
    cfg = load_config(tmp_path / "nope.yaml", {"model": {"tau": 0.6}, "wake": {"threshold": 0.7}})
    assert cfg["model"]["tau"] == 0.6 and cfg["wake"]["threshold"] == 0.7 and cfg["wake"]["hop_s"] == 0.25
    (tmp_path / "c.yaml").write_text("wake:\n  threshold: 0.8\nmock:\n  spotify: true\n")
    cfg = load_config(tmp_path / "c.yaml")
    assert cfg["wake"]["threshold"] == 0.8 and cfg["mock"]["spotify"] is True and cfg["mock"]["light"] is True


def test_parse_mock():
    assert parse_mock(None) == {}
    assert parse_mock("all") == {"light": True, "aircon": True, "spotify": True}
    assert parse_mock("none") == {"light": False, "aircon": False, "spotify": False}
    assert parse_mock("spotify") == {"spotify": True}
    assert parse_mock("lights, aircon") == {"light": True, "aircon": True}
    with pytest.raises(ValueError):
        parse_mock("toaster")


# ---------------------------------------------------------------- understood-as text
def test_rebuild_text_for_every_command_and_slot():
    for c in L.COMMANDS:
        if c in L.SLOT_VALUES:
            for v in L.SLOT_VALUES[c]:
                t = rebuild_text(c, v)
                assert v.lower() in t.lower()
        else:
            assert rebuild_text(c) and rebuild_text(c) != c
    assert rebuild_text("TIMER", "30 seconds") == "Set a timer for 30 seconds"
    assert rebuild_text("COLOR", "Red") == "Make the light red"
    assert rebuild_text("OUT_OF_SCOPE") == "(not a command)"


def test_convo_log_roundtrip(tmp_path):
    import numpy as np
    log = ConvoLog(tmp_path)
    tid = log.log_turn(Command("TIMER", "1 minute", 0.9), True, "Timer set for 1 minute.", "/replies/x.mp3",
                       audio=np.zeros(1600, np.float32))
    log.mark(tid, True)
    rec = log.last()
    assert len(rec) == 1 and rec[0]["heard"]["text"] == "Set a timer for 1 minute" and rec[0]["heard"]["slot"] == "1 minute"
    assert (tmp_path / "clips" / f"{tid}.wav").exists()
    assert "transcript" not in json.dumps(rec)


# ---------------------------------------------------------------- metrics
def test_metrics_files_and_summary(tmp_path):
    m = Metrics(model_info={"name": "x"}, root=str(tmp_path))
    for i in range(3):
        t = Turn()
        for name, ts in [("wake_start", 0.0), ("wake_done", 0.016), ("chime", 0.02), ("capture_start", 0.02),
                         ("speech_end", 1.0), ("window_closed", 1.7), ("prep_done", 1.71), ("vcm_done", 1.78),
                         ("actuator_start", 1.781), ("actuator_done", 1.80), ("reply_start", 1.85)]:
            t.mark(name, ts)
        m.record_turn(f"t{i}", t, command="TIME", prob=0.9, accepted=True, reply_key="time_now")
    m.apply_mark("t0", True)
    m.apply_mark("t1", False, "WEATHER")
    m.note_wake(True)
    m.note_wake(False)
    m.flush()
    d = Path(m.dir)
    assert {p.name for p in d.iterdir()} >= {"turns.csv", "resources.csv", "summary.json", "report.md"}
    s = json.loads((d / "summary.json").read_text())
    assert s["n_turns"] == 3 and s["marked_accuracy"] == 0.5 and s["false_wakes"] == 1 and s["wake_count"] == 2
    st = s["stages_ms"]
    assert st["vcm_infer"]["p50"] == pytest.approx(70, abs=1) and st["end_to_end"]["p50"] == pytest.approx(150, abs=1)
    assert st["speech_end_to_reply"]["p50"] == pytest.approx(850, abs=1)
    assert st["wake_infer"]["p50"] == pytest.approx(16, abs=1)
    header = (d / "turns.csv").read_text().splitlines()[0]
    for col in ("vcm_infer", "end_to_end", "mark_correct", "command"):
        assert col in header
    snap = m.snapshot()
    assert snap["type"] == "metrics" and snap["stages"]["vcm_infer"]["last"] == pytest.approx(70, abs=1)
    assert snap["end_to_end"]["p50"] and snap["false_wakes"] == 1 and snap["model"] == {"name": "x"}
    assert "rss_mb" in (d / "resources.csv").read_text().splitlines()[0]


# ---------------------------------------------------------------- dashboard server
def test_dashboard_websocket_roundtrip(tmp_path):
    from aiohttp.test_utils import TestClient, TestServer
    from runtime.bus import Bus
    from runtime.state import State
    from runtime.ui_server import build_app

    async def go():
        bus, state = Bus(), State()
        convo = ConvoLog(tmp_path)
        convo.log_turn(Command("TIME"), True, "It's 4:00 PM.")
        app = build_app(bus, state, convo, tmp_path / "replies")
        async with TestClient(TestServer(app)) as client:
            r = await client.get("/")
            assert r.status == 200 and "Understood as" in await r.text()
            assert (await client.get("/static/app.js")).status == 200
            assert (await client.get("/static/fonts/Inter-Variable.woff2")).status == 200
            ws = await client.ws_connect("/ws")
            first = json.loads((await ws.receive()).data)
            assert first["type"] == "state" and "music" in first["state"] and "thermostat" in first["state"]
            turn = json.loads((await ws.receive()).data)
            assert turn["type"] == "turn" and turn["heard"]["text"] == "What time is it?"
            bus.publish({"type": "heard", "command": "TIME", "accepted": True})
            assert json.loads((await ws.receive()).data)["type"] == "heard"
            await ws.send_str(json.dumps({"type": "command", "command": "LIGHT_ON"}))
            msg = await asyncio.wait_for(bus.inbound.get(), 2)
            assert msg == {"type": "command", "command": "LIGHT_ON"}
            await ws.close()
    asyncio.run(go())


def test_dashboard_has_no_leftover_old_labels():
    html = (ROOT / "runtime" / "ui" / "index.html").read_text()
    js = (ROOT / "runtime" / "ui" / "app.js").read_text()
    assert "transcript" not in html.lower()                      # the log says "Understood as", never "transcript"
    assert "Understood as" in html
    for old in ("speaker-dot", "aircon-setpoint", "intent_p", "LIGHT_ADJUST"):
        assert old not in html + js
    ids = set(__import__("re").findall(r'getElementById\("([^"]+)"\)', js))
    html_ids = set(__import__("re").findall(r'id="([^"]+)"', html))
    assert not (ids - html_ids - {"diag-model-name", "diag-model-params", "diag-model-version"}) , ids - html_ids


# ---------------------------------------------------------------- spotify_auth helper
def test_spotify_auth_helpers():
    sys.path.insert(0, str(ROOT / "scripts"))
    import spotify_auth as A
    url = A.authorize_url("cid", "http://127.0.0.1:8888/callback", "st")
    assert url.startswith("https://accounts.spotify.com/authorize?")
    assert "user-modify-playback-state" in url and "user-read-playback-state" in url
    assert "redirect_uri=http%3A%2F%2F127.0.0.1%3A8888%2Fcallback" in url and "response_type=code" in url
    assert A.extract_code("http://127.0.0.1:8888/callback?code=XYZ&state=st", "st") == "XYZ"
    assert A.extract_code("code=XYZ&state=st") == "XYZ"
    assert A.extract_code("XYZ") == "XYZ"
    with pytest.raises(SystemExit):
        A.extract_code("http://127.0.0.1:8888/callback?code=XYZ&state=other", "st")
    with pytest.raises(SystemExit):
        A.extract_code("http://127.0.0.1:8888/callback?error=access_denied")
