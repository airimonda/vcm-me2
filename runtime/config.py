"""Runtime configuration: DEFAULTS < config/runtime.yaml < CLI overrides."""
from __future__ import annotations

import copy
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config" / "runtime.yaml"

DEFAULTS: dict = {
    "model": {"path": "models/vcm_conformer_M_ens3.onnx", "tau": None, "intra_threads": 2},
    "wake": {"enabled": True, "path": "models/wake/vcm_wake_int8.onnx", "threshold": 0.55, "consecutive": 1,
             "hop_s": 0.25, "cooldown_s": 1.0, "intra_threads": 1},
    "audio": {"backend": "auto", "device": None, "block_ms": 100, "play": True, "earcon_volume": 0.15},
    "capture": {"max_s": 5.0, "min_s": 0.8, "end_silence_s": 0.7, "no_speech_timeout_s": 3.0,
                "start_ignore_s": 0.25, "margin_db": 10.0, "min_db": -55.0, "min_speech_s": 0.12},
    "dispatcher": {"oos_reply": "sorry", "ducking": True, "duck_to": 20},
    "mock": {"light": True, "aircon": True, "spotify": False},
    "music": {"credentials": "config/spotify.json", "device_name": "Watson", "context_uri": "liked",
              "shuffle": True, "volume_step": 15, "poll_s": 5.0, "timeout_s": 4.0},
    "comms": {"contact": "Mom", "message_text": "I'm on my way"},
    "thermostat": {"min": 16, "max": 30, "start_room_temp": 28.0, "drift_period_s": 120},
    "weather": {"city": "Quezon City", "latitude": 14.676, "longitude": 121.0437, "timeout_s": 3.0},
    "clock": {"timezone": None},
    "replies": {"dir": "replies", "catalogue": "config/replies.json"},
    "ui": {"enabled": True, "host": "0.0.0.0", "port": 8080},
    "metrics": {"dir": "runtime_logs", "live_log": None},
}

MOCK_KEYS = ("light", "aircon", "spotify")


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | os.PathLike | None = None, overrides: dict | None = None) -> dict:
    """Load YAML (missing file = defaults) and apply `overrides` (nested dict) on top."""
    cfg = copy.deepcopy(DEFAULTS)
    p = Path(path) if path else DEFAULT_CONFIG
    if p.exists():
        import yaml
        cfg = deep_merge(cfg, yaml.safe_load(p.read_text()) or {})
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def parse_mock(spec: str | None) -> dict:
    """--mock value -> {light, aircon, spotify} overrides. "all", "none", or a comma list of devices to
    force to mock (aliases: lights, thermostat/ac, music); unlisted devices keep their configured value."""
    if spec is None:
        return {}
    spec = spec.strip().lower()
    if spec == "all":
        return {k: True for k in MOCK_KEYS}
    if spec in ("", "none"):
        return {k: False for k in MOCK_KEYS}
    alias = {"lights": "light", "thermostat": "aircon", "ac": "aircon", "music": "spotify"}
    out = {}                                  # only the listed devices change; the rest keep config/runtime.yaml
    for tok in spec.split(","):
        tok = alias.get(tok.strip(), tok.strip())
        if tok in MOCK_KEYS:
            out[tok] = True
        elif tok:
            raise ValueError(f"unknown --mock device {tok!r} (choose from {', '.join(MOCK_KEYS)}, all, none)")
    return out


def resolve(path: str | os.PathLike | None) -> Path | None:
    """repo-relative path -> absolute (None stays None)"""
    if path is None:
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p
