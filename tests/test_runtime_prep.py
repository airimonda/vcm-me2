"""runtime.prep is the SAME preparation scripts/pack_data.py applies to the test/holdout clips."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
HOLDOUT = Path.home() / "ai231-me2-collated" / "dataset" / "holdout"
PACKS = ROOT / "data" / "packs"


def test_pack_data_uses_the_shared_helpers():
    import pack_data
    from runtime import prep
    assert pack_data.trim_speech is prep.trim_speech
    assert pack_data.best_window is prep.best_window
    assert pack_data.centre_pad is prep.centre_pad
    assert pack_data.WIN_TEST == prep.WINDOW == 80_000


@pytest.mark.skipif(not (HOLDOUT.exists() and (PACKS / "holdout_meta.parquet").exists()), reason="holdout data missing")
def test_prepare_window_reproduces_the_packed_holdout_clips_exactly():
    import pandas as pd
    import soundfile as sf
    from runtime.prep import prepare_window
    meta = pd.read_parquet(PACKS / "holdout_meta.parquet")
    pack = np.load(PACKS / "holdout_audio.npy", mmap_mode="r")
    n = 0
    for i in range(0, len(meta), 7):
        path = HOLDOUT / meta.file.iloc[i]
        if not path.exists():
            continue
        x, sr = sf.read(path, dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != 16000:
            continue
        assert np.array_equal(prepare_window(x), pack[i].astype(np.float32) / 32768.0), meta.file.iloc[i]
        n += 1
    assert n >= 10


def test_runtime_prep_needs_no_heavy_imports():
    code = ("import sys, runtime.prep, runtime.endpoint, runtime.wake, runtime.model, runtime.dispatcher, runtime.devices;"
            "bad=[m for m in ('torch','pandas','scipy','vcm') if m in sys.modules];"
            "print('BAD', bad) if bad else print('OK')")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT)
    assert r.stdout.strip() == "OK", r.stdout + r.stderr
