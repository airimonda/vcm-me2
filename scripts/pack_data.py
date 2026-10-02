"""Decode a dataset split into int16 arrays + a meta parquet (fast loading for training).

Per clip: energy-based speech trim (frame energy relative to the loudest frame and to
the estimated noise floor, small pre/post pad), then

  test / holdout : speech centred in exactly 5.0 s (80,000 samples); if the trimmed speech is
                   longer, the highest-energy 5 s is kept.
  train          : trimmed speech kept up to 6.0 s (highest-energy 6 s if longer) and centred in
                   a 6.0 s buffer (96,000 samples), zero padded. Training takes a random 5 s
                   crop from this buffer (random time shift) so shifts never run out of audio.
                   Default un-shifted window is samples [8000, 88000).

Outputs in --out (default data/packs):
  <split>_audio.npy   int16 (N, 96000 or 80000)
  <split>_meta.parquet  labels (cmd_idx, slot_head_idx, slot_value_idx, variation_idx),
                        metadata, trimmed length
  neg_train_* / neg_test_*  (--negatives) same format as train / test, from the synthetic negatives
  noise.npz           (--noise) all noise wavs concatenated int16 + offsets

Usage:
  python scripts/pack_data.py --dataset ~/ai231-me2-collated/dataset --splits train test holdout
  python scripts/pack_data.py --noise data/noise
  python scripts/pack_data.py --negatives ~/ai231-me2-collated/dataset/synthetic_negatives   # neg_train, neg_test
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vcm.labels import row_to_labels  # noqa: E402

SR = 16000
WIN_TEST = 5 * SR          # 80,000
BUF_TRAIN = 6 * SR         # 96,000
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
    out = np.zeros(length, dtype=np.int16)
    s = (length - len(x)) // 2
    out[s: s + len(x)] = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
    return out, len(x)


def process(args):
    path, length = args
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = resample_poly(x, SR, sr).astype(np.float32)
    x = best_window(trim_speech(x), length)
    return centre_pad(x, length) + (len(x) / SR,)


def pack_split(dataset: Path, split: str, out: Path, workers: int, name: str | None = None):
    """Pack <dataset>/<split>/ as <out>/<name>_audio.npy + <name>_meta.parquet (name defaults to split).
    The buffer is the train-style 96,000 samples for split == "train", else the 80,000 test window."""
    name = name or split
    df = pd.read_csv(dataset / split / "manifest.csv")
    length = BUF_TRAIN if split == "train" else WIN_TEST
    paths = [(str(dataset / split / f), length) for f in df["file"]]
    arr = np.lib.format.open_memmap(out / f"{name}_audio.npy", mode="w+", dtype=np.int16,
                                    shape=(len(df), length))
    lens = np.zeros(len(df), dtype=np.float32)
    with ProcessPoolExecutor(workers) as ex:
        for i, (a, _, d) in enumerate(tqdm(ex.map(process, paths, chunksize=32), total=len(df), desc=split)):
            arr[i] = a
            lens[i] = d
    arr.flush()
    lab = np.array([row_to_labels(r) for _, r in df.iterrows()], dtype=np.int16)
    meta = pd.DataFrame({
        "file": df["file"], "transcript": df["transcript"], "command": df["command"],
        "variation": df["variation"], "slot_value": df["slot_value"], "bucket": df["bucket"],
        "speaker_id": df["speaker_id"].astype(str), "source": df["source"],
        "is_synthetic": df["is_synthetic"].astype(int), "accent_group": df["accent_group"],
        "duration_s": df["duration_s"], "speech_s": lens,
        "cmd_idx": lab[:, 0], "slot_head_idx": lab[:, 1], "slot_value_idx": lab[:, 2],
        "variation_idx": lab[:, 3],
    })
    for extra in ("neg_kind", "source_files"):       # synthetic-negative manifests carry these
        if extra in df.columns:
            meta[extra] = df[extra].fillna("").astype(str)
    meta.to_parquet(out / f"{name}_meta.parquet", index=False)
    print(f"{name}: {len(df)} clips, buffer {length}, mean speech {lens.mean():.2f}s, "
          f"clipped-to-window {(lens >= length / SR - 1e-3).sum()}")


def pack_noise(noise_dir: Path, out: Path):
    chunks, offs = [], [0]
    for f in sorted(noise_dir.glob("*.wav")):
        x, sr = sf.read(str(f), dtype="float32")
        if x.ndim > 1:
            x = x.mean(1)
        if sr != SR:
            x = resample_poly(x, SR, sr).astype(np.float32)
        chunks.append(np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16))
        offs.append(offs[-1] + len(x))
    np.savez(out / "noise.npz", audio=np.concatenate(chunks), offsets=np.array(offs, dtype=np.int64))
    print(f"noise: {len(chunks)} files, {offs[-1] / SR / 3600:.2f} h")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="~/ai231-me2-collated/dataset")
    ap.add_argument("--splits", nargs="*", default=[])
    ap.add_argument("--out", default="data/packs")
    ap.add_argument("--noise", default=None, help="folder of noise wavs to pack into noise.npz")
    ap.add_argument("--negatives", default=None,
                    help="synthetic-negatives folder (scripts/make_negatives.py output): packs <dir>/train as "
                         "neg_train (96,000-sample train buffer) and <dir>/test as neg_test (80,000 window)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for s in a.splits:
        assert s in ("train", "test", "holdout"), f"unknown split {s} (numerals is not used)"
        pack_split(Path(a.dataset).expanduser(), s, out, a.workers)
    if a.negatives:
        neg = Path(a.negatives).expanduser()
        for src, name in (("train", "neg_train"), ("test", "neg_test")):
            pack_split(neg, src, out, a.workers, name=name)
    if a.noise:
        pack_noise(Path(a.noise), out)


if __name__ == "__main__":
    main()
