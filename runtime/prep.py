"""Command-window preparation, shared by training-data packing and the live runtime.

These are the exact functions scripts/pack_data.py uses to build the test/holdout packs
(energy trim -> highest-energy 5 s -> centre in 5.0 s -> int16 round trip), so a clip captured
on the device reaches the model looking like the clips it was evaluated on. numpy only (the Pi
venv has no pandas/scipy/torch). scripts/pack_data.py imports from here; tests/runtime/test_prep.py
checks the two stay identical.
"""
from __future__ import annotations

import numpy as np

SR = 16000
WINDOW = 5 * SR            # 80,000 samples: the model input
FRAME, HOP = 320, 160      # 20 ms / 10 ms
PAD_PRE, PAD_POST = int(0.10 * SR), int(0.15 * SR)


def trim_speech(x: np.ndarray) -> np.ndarray:
    """Energy trim. x float32 in [-1,1]. Returns the speech region plus a small pad."""
    n = len(x)
    if n < FRAME * 2:
        return x
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    starts = np.arange(0, n - FRAME + 1, HOP)
    e = (c[starts + FRAME] - c[starts]) / FRAME
    db = 10 * np.log10(e + 1e-12)
    peak = db.max()
    if peak < -75:                                    # essentially silent: keep as is
        return x
    thr = max(peak - 35.0, np.percentile(db, 10) + 10.0)
    thr = min(thr, peak - 15.0)
    act = np.nonzero(db >= thr)[0]
    if len(act) == 0:
        return x
    s = max(starts[act[0]] - PAD_PRE, 0)
    t = min(starts[act[-1]] + FRAME + PAD_POST, n)
    return x[s:t]


def best_window(x: np.ndarray, length: int) -> np.ndarray:
    """Highest-energy contiguous window of `length` samples."""
    if len(x) <= length:
        return x
    c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    e = c[length:] - c[:-length]
    s = int(np.argmax(e))
    return x[s: s + length]


def centre_pad(x: np.ndarray, length: int) -> tuple[np.ndarray, int]:
    """Centre float audio in `length` zeros and quantise to int16. Returns (int16 buffer, n speech samples)."""
    out = np.zeros(length, dtype=np.int16)
    s = (length - len(x)) // 2
    out[s: s + len(x)] = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
    return out, len(x)


def prepare_window(x: np.ndarray, length: int = WINDOW) -> np.ndarray:
    """Captured mono float32 audio at 16 kHz -> float32 (length,) model input in [-1, 1].

    Same steps as scripts/pack_data.py `process` for test clips, followed by the int16 -> float
    conversion the evaluation code applies (vcm.data.to_float: /32768)."""
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    x = best_window(trim_speech(x), length)
    buf, _ = centre_pad(x, length)
    return buf.astype(np.float32) / 32768.0
