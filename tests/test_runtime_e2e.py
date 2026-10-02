"""End to end through the real ensemble ONNX: WAV -> endpointing -> window prep -> model -> dispatcher -> reply.

The wake word is mocked (the file is treated as already woken) except in the sliding-wake test, which uses a
scripted scorer. Clips are real recordings from the held-out set (~/ai231-me2-collated/dataset/holdout, read only)."""
import asyncio
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from runtime import audio as A
from runtime.app import Runtime
from runtime.config import load_config

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = Path.home() / "ai231-me2-collated" / "dataset" / "holdout" / "audio"

# (file, command, slot, reply key)
CLIPS = [
    ("holdout_000006_real_voice_WEATHER_V1_t2.wav", "WEATHER", None, "weather"),
    ("holdout_000012_real_voice_TIME_V1_t1.wav", "TIME", None, "time_now"),
    ("holdout_000020_real_voice_LIGHT_ON_V2_t2.wav", "LIGHT_ON", None, "lights_on"),
    ("holdout_000030_real_voice_PAUSE_V1_t1.wav", "PAUSE", None, "music_paused"),
    ("holdout_000048_real_voice_VOLUME_UP_V1_t2.wav", "VOLUME_UP", None, "music_volume_up"),
    ("holdout_000072_real_voice_LIST_REMINDERS_V1_t2.wav", "LIST_REMINDERS", None, "reminder_list_empty"),
    ("holdout_000080_real_voice_TIMER_V1_2_t1.wav", "TIMER", "30 seconds", "timer_set"),
    ("holdout_000098_real_voice_ALARM_V1_2_t2.wav", "ALARM", "8:00 AM", "alarm_set"),
    ("holdout_000116_real_voice_TEMPERATURE_V1_2_t1.wav", "TEMPERATURE", "22 degrees", "thermostat_set"),
    ("holdout_000134_real_voice_BRIGHTNESS_V1_2_t1.wav", "BRIGHTNESS", "60 percent", "light_brightness"),
    ("holdout_000154_real_voice_COLOR_V2_3_t2.wav", "COLOR", "Green", "light_color"),
    ("holdout_000170_real_voice_CREATE_REMINDER_V1_2_t2.wav", "CREATE_REMINDER", "Study", "reminder_saved"),
    ("holdout_000186_CommonVoice_en_common_voice_en_36827091.wav", "OUT_OF_SCOPE", None, "oos_sorry"),
]
pytestmark = pytest.mark.skipif(not HOLDOUT.exists(), reason="holdout audio not available")


@pytest.fixture(scope="module")
def runtime(tmp_path_factory):
    root = tmp_path_factory.mktemp("rt")
    cfg = load_config(None, {"mock": {"spotify": True}, "audio": {"play": False}, "ui": {"enabled": False},
                             "metrics": {"dir": str(root)}, "thermostat": {"drift_period_s": 3600}})
    loop = asyncio.new_event_loop()
    rt = loop.run_until_complete(_make(cfg))
    rt.actuators["info"]._fetch = lambda: (30, "cloudy")
    yield rt, loop
    loop.run_until_complete(rt.stop())
    loop.close()


async def _make(cfg):
    return Runtime(cfg, load_wake=True)


def play(rt, loop, wav, **kw):
    before = len(rt.metrics.turns)
    loop.run_until_complete(rt.run_stream(A.WavSource(wav, **kw), use_wake=False))
    rows = rt.metrics.turns[before:]
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize("fname,command,slot,reply", CLIPS)
def test_real_clip_reaches_the_expected_command_and_reply(runtime, fname, command, slot, reply):
    rt, loop = runtime
    if command == "PAUSE":
        play(rt, loop, A.load_wav(HOLDOUT / CLIPS[2][0]))       # any command first; PAUSE then runs on the mock player
    row = play(rt, loop, A.load_wav(HOLDOUT / fname))
    assert row["command"] == command and (row["slot"] or None) == slot, row
    assert row["reply_key"] == reply
    assert row["accepted"] == (command != "OUT_OF_SCOPE")
    for stage in ("capture", "prep", "vcm_infer", "decode_dispatch", "actuator", "end_to_end", "speech_end_to_reply"):
        assert row[stage] >= 0, stage
    assert row["vcm_infer"] < 2000


def test_device_state_after_the_clips(runtime):
    rt, _ = runtime
    d = rt.state.data
    assert d["light"]["on"] and d["light"]["color_name"] == "Green" and d["light"]["brightness"] == 60
    assert d["thermostat"]["setpoint"] == 22 and d["thermostat"]["on"]
    assert any(t["label"] == "30 seconds" for t in d["timers"])
    assert any(a["time"] == "8:00" and a["ampm"] == "am" for a in d["alarms"])
    assert d["weather"]["temp"] == 30
    assert "Study" in [r["topic"] for r in d["reminders"]]


def test_tau_override_turns_a_confident_command_into_oos(runtime):
    rt, loop = runtime
    wav = A.load_wav(HOLDOUT / CLIPS[1][0])
    old = rt.model.tau
    try:
        rt.model.tau = 0.999
        row = play(rt, loop, wav)
        assert row["command"] == "OUT_OF_SCOPE" and row["prob"] > 0.5
    finally:
        rt.model.tau = old


def test_scripted_wake_then_command_in_a_noisy_stream(runtime):
    """noise -> 'Watson' (a tone the scripted scorer recognises) -> pause -> real command -> silence"""
    rt, loop = runtime
    sr = 16000
    rng = np.random.RandomState(1)
    noise = lambda s: (rng.randn(int(s * sr)) * 10 ** (-58 / 20)).astype(np.float32)   # noqa: E731
    t = np.arange(int(0.5 * sr)) / sr
    watson = (np.sin(2 * np.pi * 1200 * t) * 0.2).astype(np.float32)
    cmd = A.load_wav(HOLDOUT / CLIPS[6][0])                                              # timer 30 seconds
    stream = np.concatenate([noise(1.5), watson, noise(0.5), cmd, noise(0.5)])

    class ScriptedWake:
        window = 24000

        def timed_score(self, a):
            # P(WATSON) = 1 when the last 0.25 s is the 1200 Hz tone
            seg = a[-4000:]
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
            peak_hz = np.argmax(spec) * sr / len(seg)
            return (0.95 if abs(peak_hz - 1200) < 60 and np.abs(seg).max() > 0.1 else 0.02), 0.5
    real, rt.wake_model = rt.wake_model, ScriptedWake()
    try:
        before, wakes = len(rt.metrics.turns), rt.metrics.wake_count
        loop.run_until_complete(rt.run_stream(A.WavSource(stream), use_wake=True))
        rows = rt.metrics.turns[before:]
    finally:
        rt.wake_model = real
    assert len(rows) == 1, rows
    assert rows[0]["command"] == "TIMER" and rows[0]["slot"] == "30 seconds"
    assert rows[0]["wake_infer"] == pytest.approx(0.5, abs=0.2) and rows[0]["wake_to_chime"] >= 0
    assert rt.metrics.wake_count == wakes + 1


def test_wake_without_speech_counts_as_a_false_wake(runtime):
    rt, loop = runtime
    sr = 16000
    t = np.arange(int(0.5 * sr)) / sr
    stream = np.concatenate([np.zeros(int(1.0 * sr), np.float32), (np.sin(2 * np.pi * 1200 * t) * 0.2).astype(np.float32),
                             np.zeros(int(4.0 * sr), np.float32)])

    class ScriptedWake:
        window = 24000

        def timed_score(self, a):
            return (0.95 if np.abs(a[-4000:]).max() > 0.1 else 0.0), 0.1
    real, rt.wake_model = rt.wake_model, ScriptedWake()
    try:
        fw, before = rt.metrics.false_wakes, len(rt.metrics.turns)
        loop.run_until_complete(rt.run_stream(A.WavSource(stream), use_wake=True))
        rows = rt.metrics.turns[before:]
    finally:
        rt.wake_model = real
    assert [r["command"] for r in rows] == ["NO_SPEECH"] and rows[0]["reason"] == "no_speech"
    assert rt.metrics.false_wakes == fw + 1


def test_ducking_brackets_every_turn(runtime):
    rt, loop = runtime
    player = rt.music.player
    player.playing, player.started, player.volume = True, True, 60
    rt.music.player.depth = 0
    play(rt, loop, A.load_wav(HOLDOUT / CLIPS[1][0]))
    loop.run_until_complete(asyncio.sleep(0.05))
    rt.music._pool.submit(lambda: None).result()
    assert player.volume == 60 and player.pre_duck is None and player.depth == 0


def test_cli_input_wav_prints_json_results(tmp_path):
    files = [str(HOLDOUT / CLIPS[1][0]), str(HOLDOUT / CLIPS[8][0])]
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "pi_runtime.py"), "--mock", "all", "--input-wav", *files,
                        "--json", "--metrics-dir", str(tmp_path)], capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    rows = [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]
    assert [(x["command"], x["slot"]) for x in rows] == [("TIME", None), ("TEMPERATURE", "22 degrees")]
    runs = list(tmp_path.glob("2*"))
    assert len(runs) == 1 and (runs[0] / "turns.csv").exists() and (runs[0] / "summary.json").exists()
    assert json.loads((runs[0] / "summary.json").read_text())["n_turns"] == 2
