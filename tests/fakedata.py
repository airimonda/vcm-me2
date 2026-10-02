"""Tiny fake dataset (gold-style layout) for testing make_negatives / pack_data / train without real data."""
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
COLS = ["file", "transcript", "command", "variation", "slot_value", "bucket", "speaker_id", "source", "is_synthetic",
        "accent_group", "duration_s", "out_of_scope"]
# (command, variation phrase, slot value, words in transcript)
PHRASES = [("PLAY_MUSIC", "Play some music", ""), ("WEATHER", "Tell me the weather", ""),
           ("TIMER", "Timer 10 seconds", "10 seconds"), ("PAUSE", "Pause", ""), ("TIME", "What time is it?", ""),
           ("LIGHT_ON", "Power on the lights", ""), ("STOP", "Stop", ""), ("TIMER", "Timer 1 minute", "1 minute")]


def speechlike(rng, dur_s: float, f0: float, level_db: float = -24.0) -> np.ndarray:
    n = int(dur_s * SR)
    t = np.arange(n) / SR
    x = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 6))
    env = np.zeros(n)
    nb = 4
    lead = int(0.15 * SR)
    seg = (n - 2 * lead) // nb
    for b in range(nb):
        s = lead + b * seg
        env[s: s + int(seg * 0.7)] = np.hanning(int(seg * 0.7))
    x = x * env + rng.normal(0, 0.003, n)
    return (x * 10 ** (level_db / 20) / (np.sqrt(np.mean(x ** 2)) + 1e-9)).astype(np.float32)


def make_split(root: Path, split: str, n: int, rng, tag: str) -> pd.DataFrame:
    (root / split / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        spk = f"{tag}{i % 4}"
        oos = i % 5 == 4
        dur = float(rng.uniform(1.6, 3.5))
        name = f"audio/{split}_{i:03d}.wav"
        x = speechlike(rng, dur, 100 + 25 * (i % 4))
        sf.write(str(root / split / name), np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16), SR,
                 subtype="PCM_16")
        if oos:
            cmd, var, slot, tr = "OUT_OF_SCOPE", "", "", "what is the capital of france today"
        else:
            cmd, var, slot = PHRASES[i % len(PHRASES)]
            tr = var.lower()
        rows.append({"file": name, "transcript": tr, "command": cmd, "variation": var, "slot_value": slot,
                     "bucket": "OUT_OF_SCOPE" if oos else var, "speaker_id": spk, "source": "fake",
                     "is_synthetic": i % 2, "accent_group": "Synthetic" if i % 2 else "Fake", "duration_s": round(dur, 3),
                     "out_of_scope": int(oos)})
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(root / split / "manifest.csv", index=False)
    return df


def make_noise(folder: Path, n_files: int = 6, seed: int = 0) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for k in range(n_files):
        x = np.convolve(rng.normal(0, 0.05, 4 * SR), np.ones(1 + k) / (1 + k), mode="same").astype(np.float32)
        sf.write(str(folder / f"noise_{k}.wav"), x, SR, subtype="PCM_16")
    return folder


def make_fake_dataset(root: Path, n_per_split: int = 10, seed: int = 0, n_noise: int = 6):
    """Writes <root>/{train,test} (distinct speakers per split: tr*, te*) and <root>_noise/. Returns (root, noise)."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    make_split(root, "train", n_per_split, rng, "tr")
    make_split(root, "test", n_per_split, rng, "te")
    return root, make_noise(root.parent / (root.name + "_noise"), n_noise, seed)
