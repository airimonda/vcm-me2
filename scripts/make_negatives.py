"""Generate a SEPARATE folder of synthetic out-of-scope negatives (files on disk, made once).

    python scripts/make_negatives.py --dataset ~/ai231-me2-collated/dataset --noise data/noise \
        --out ~/ai231-me2-collated/dataset/synthetic_negatives --n-train 1000 --n-test 250 --seed 0

Input  : <dataset>/{train,test}/manifest.csv + audio/*.wav (16 kHz mono 16-bit), and a folder of noise wavs.
Output : <out>/{train,test}/manifest.csv + audio/*.wav, <out>/summary.json, <out>/README.md.

Every output row is OUT_OF_SCOPE (command=OUT_OF_SCOPE, out_of_scope=1, bucket=OUT_OF_SCOPE,
source=synthetic_negative, is_synthetic=1, accent_group=Synthetic). Extra columns: `neg_kind`
and `source_files` (";"-joined: "<split>/<file>" for gold clips, "noise/<name>" for noise files).

Train negatives are built ONLY from the gold train split, test negatives ONLY from the gold test
split (speakers stay disjoint). Noise files are split by filename hash: 80% train / 20% test.

Kinds (equal shares, each clip 1-5 s):
  noise_only   a noise segment at a random level (-45 .. -15 dBFS RMS)
  babble       3-6 clips of the split (different speakers where possible) summed at random offsets,
               overall level of normal speech (-30 .. -20 dBFS RMS)
  reversed     one clip of the split played backwards
  truncated    first 25-40% of the speech of an in-scope clip with >= 3 words and >= 1.5 s, so a full
               command is never present; optionally with light noise
  near_silence very low noise (-60 .. -50 dBFS) or digital near-silence with mic hiss (~0.4-2 LSB)

Deterministic: same inputs + same --seed give byte-identical wavs and manifests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pack_data import trim_speech  # noqa: E402

SR = 16000
KINDS = ["noise_only", "babble", "reversed", "truncated", "near_silence"]
MIN_S, MAX_S = 1.0, 5.0
REQUIRED_COLS = ["file", "transcript", "command", "variation", "slot_value", "bucket", "speaker_id", "source",
                 "is_synthetic", "accent_group", "duration_s", "out_of_scope"]
EXTRA_COLS = ["neg_kind", "source_files"]
OOS = "OUT_OF_SCOPE"


# ------------------------------------------------------------------ small helpers
def rms_db(x: np.ndarray) -> float:
    return 10 * np.log10(float(np.mean(np.square(x, dtype=np.float64))) + 1e-20)


def set_rms(x: np.ndarray, db: float) -> np.ndarray:
    cur = rms_db(x)
    if cur < -120:                                  # digital silence: nothing to scale
        return x
    y = x * 10 ** ((db - cur) / 20)
    peak = float(np.abs(y).max())
    return y * (0.98 / peak) if peak > 0.98 else y


def read_mono(path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = resample_poly(x, SR, sr).astype(np.float32)
    return x


def noise_split(files: list[Path]) -> tuple[list[Path], list[Path]]:
    """Deterministic 80/20 split by md5 of the file name. Guarantees both sides are non-empty when
    there are >= 2 files (the file with the lowest/highest hash moves over if one side would be empty)."""
    h = {f: int(hashlib.md5(f.name.encode()).hexdigest(), 16) for f in files}
    tr = sorted((f for f in files if h[f] % 100 < 80), key=lambda f: f.name)
    te = sorted((f for f in files if h[f] % 100 >= 80), key=lambda f: f.name)
    if len(files) >= 2:
        if not te:
            mv = max(tr, key=lambda f: (h[f], f.name))
            tr.remove(mv)
            te.append(mv)
        elif not tr:
            mv = min(te, key=lambda f: (h[f], f.name))
            te.remove(mv)
            tr.append(mv)
    return tr, te


def noise_segment(path: Path, n: int, rng) -> np.ndarray:
    """n samples at 16 kHz from a random place in the file (tiled if the file is shorter)."""
    info = sf.info(str(path))
    need = int(np.ceil(n * info.samplerate / SR)) + 1
    if info.frames > need:
        start = int(rng.integers(0, info.frames - need + 1))
        x, sr = sf.read(str(path), start=start, frames=need, dtype="float32", always_2d=False)
    else:
        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(1)
    if sr != SR:
        x = resample_poly(x, SR, sr).astype(np.float32)
    if len(x) < n:
        x = np.tile(x if len(x) else np.zeros(1, np.float32), int(np.ceil(n / max(len(x), 1))))
    return x[:n].astype(np.float32)


def safe(s) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in str(s))


class Gold:
    """One gold split: manifest rows + lazy wav reads."""

    def __init__(self, root: Path, split: str):
        self.root, self.split = root, split
        self.df = pd.read_csv(root / split / "manifest.csv")
        self.n = len(self.df)
        self.speaker = self.df["speaker_id"].astype(str).to_numpy()
        words = self.df["transcript"].fillna("").astype(str).str.split().str.len().to_numpy()
        dur = pd.to_numeric(self.df["duration_s"], errors="coerce").fillna(0).to_numpy()
        flag = (pd.to_numeric(self.df["out_of_scope"], errors="coerce").fillna(0).to_numpy()
                if "out_of_scope" in self.df.columns else np.zeros(self.n))
        inscope = (self.df["command"].astype(str) != OOS).to_numpy() & (flag != 1)
        self.trunc_pool = np.nonzero(inscope & (words >= 3) & (dur >= 1.5))[0]

    def path(self, i):
        return self.root / self.split / self.df["file"].iloc[i]

    def ref(self, i):
        return f"{self.split}/{self.df['file'].iloc[i]}"

    def speech(self, i):
        return trim_speech(read_mono(self.path(i)))


# ------------------------------------------------------------------ the five generators
# each returns (audio float32, [source refs], first source speaker)
def gen_noise_only(rng, gold: Gold, noise: list[Path]):
    n = int(rng.uniform(MIN_S, MAX_S) * SR)
    f = noise[int(rng.integers(len(noise)))]
    x = set_rms(noise_segment(f, n, rng), float(rng.uniform(-45, -15)))
    return x, [f"noise/{f.name}"], f.stem


def gen_babble(rng, gold: Gold, noise: list[Path]):
    D = int(rng.uniform(MIN_S, MAX_S) * SR)
    k = int(rng.integers(3, 7))
    chosen, seen = [], set()
    perm = rng.permutation(gold.n)
    for i in perm:                                   # prefer distinct speakers
        if gold.speaker[i] not in seen:
            chosen.append(int(i))
            seen.add(gold.speaker[i])
            if len(chosen) == k:
                break
    for i in perm:                                   # tiny datasets: fill up with other clips
        if len(chosen) >= min(k, gold.n):
            break
        if int(i) not in chosen:
            chosen.append(int(i))
    mix = np.zeros(D, np.float32)
    for i in chosen:
        x = gold.speech(i)
        if len(x) > D:
            s = int(rng.integers(0, len(x) - D + 1))
            x = x[s: s + D]
        x = x * 10 ** (float(rng.uniform(-6, 0)) / 20)
        off = int(rng.integers(0, D - len(x) + 1))
        mix[off: off + len(x)] += x
    mix = set_rms(mix, float(rng.uniform(-30, -20)))
    return mix, [gold.ref(i) for i in chosen], gold.speaker[chosen[0]]


def gen_reversed(rng, gold: Gold, noise: list[Path]):
    x, i = None, None
    for _ in range(40):
        i = int(rng.integers(gold.n))
        x = gold.speech(i)
        if len(x) >= MIN_S * SR:
            break
    if len(x) < MIN_S * SR:                          # nothing long enough: pad with faint hiss
        x = np.concatenate([x, rng.normal(0, 10 ** (-60 / 20), int(MIN_S * SR) - len(x))]).astype(np.float32)
    x = x[::-1]
    if len(x) > MAX_S * SR:
        s = int(rng.integers(0, len(x) - int(MAX_S * SR) + 1))
        x = x[s: s + int(MAX_S * SR)]
    return np.ascontiguousarray(x, dtype=np.float32), [gold.ref(i)], gold.speaker[i]


def gen_truncated(rng, gold: Gold, noise: list[Path]):
    if len(gold.trunc_pool) == 0:
        raise SystemExit(f"{gold.split}: no in-scope clip with >= 3 words and duration >= 1.5 s for 'truncated'")
    i = int(gold.trunc_pool[int(rng.integers(len(gold.trunc_pool)))])
    x = gold.speech(i)
    keep = min(int(rng.uniform(0.25, 0.40) * len(x)), int(MAX_S * SR))
    keep = max(keep, 1)
    seg = x[:keep].copy()
    fade = min(int(0.015 * SR), keep)
    seg[keep - fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
    D = int(np.clip(keep + rng.uniform(0.2, 1.5) * SR, MIN_S * SR, MAX_S * SR))
    D = max(D, keep)
    buf = np.zeros(D, np.float32)
    if rng.random() < 0.5 and noise:                 # light noise
        f = noise[int(rng.integers(len(noise)))]
        buf += set_rms(noise_segment(f, D, rng), float(rng.uniform(-48, -35)))
        src = [gold.ref(i), f"noise/{f.name}"]
    else:
        src = [gold.ref(i)]
    off = int(rng.integers(0, D - keep + 1))
    buf[off: off + keep] += seg
    return np.clip(buf, -1, 1), src, gold.speaker[i]


def gen_near_silence(rng, gold: Gold, noise: list[Path]):
    n = int(rng.uniform(MIN_S, MAX_S) * SR)
    if rng.random() < 0.5:                           # very low noise
        x = set_rms(rng.normal(0, 1, n).astype(np.float32), float(rng.uniform(-60, -50)))
    else:                                            # digital near-silence: mostly 0 / +-1 LSB mic hiss
        lsb = float(rng.uniform(0.4, 2.0))
        x = np.round(rng.normal(0, lsb, n)).astype(np.float32) / 32768.0
    return x, [], "synthetic"


GENERATORS = {"noise_only": gen_noise_only, "babble": gen_babble, "reversed": gen_reversed,
              "truncated": gen_truncated, "near_silence": gen_near_silence}


# ------------------------------------------------------------------ split driver
def kind_counts(n: int) -> dict[str, int]:
    return {k: n // len(KINDS) + (1 if j < n % len(KINDS) else 0) for j, k in enumerate(KINDS)}


def make_split(root: Path, split: str, noise: list[Path], out: Path, n: int, seed: int, columns: list[str]):
    gold = Gold(root, split)
    rng = np.random.default_rng([seed, 0 if split == "train" else 1])
    kinds = [k for k, c in kind_counts(n).items() for _ in range(c)]
    kinds = [kinds[j] for j in rng.permutation(len(kinds))]
    needs_noise = {"noise_only"} & set(kinds)
    if needs_noise and not noise:
        raise SystemExit(f"no noise files available for the {split} split")
    (out / split / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for j, kind in enumerate(kinds):
        x, src, spk = GENERATORS[kind](rng, gold, noise)
        pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype(np.int16)
        name = f"audio/neg_{kind}_{j:05d}.wav"
        sf.write(str(out / split / name), pcm, SR, subtype="PCM_16")
        row = {c: "" for c in columns}
        row.update({"file": name, "transcript": f"<{kind}>", "command": OOS, "variation": "", "slot_value": "",
                    "bucket": OOS, "speaker_id": f"neg_{kind}_{safe(spk)}", "source": "synthetic_negative",
                    "is_synthetic": 1, "accent_group": "Synthetic", "duration_s": round(len(pcm) / SR, 3),
                    "out_of_scope": 1, "neg_kind": kind, "source_files": ";".join(src)})
        rows.append(row)
    pd.DataFrame(rows, columns=columns).to_csv(out / split / "manifest.csv", index=False)
    return pd.DataFrame(rows)


README = """# Synthetic negatives

Synthetic OUT_OF_SCOPE clips (noise, babble, reversed speech, truncated commands, near-silence) used
to teach the voice command model not to fire on non-command audio ("misfires").

**This folder is NOT part of the published gold splits.** It is a separate, generated dataset:
it must not be mixed into `train/`, `test/` or `holdout/` or reported as gold test data.
`neg_test` is only a diagnostic (misfire rate); gold test numbers stay comparable with earlier runs.

* Made once by `scripts/make_negatives.py` with seed {seed} (deterministic), written to disk as 16 kHz
  mono 16-bit wavs. Layout is the same as the gold splits: `train/` and `test/`, each with
  `manifest.csv` and `audio/*.wav`.
* Train negatives use ONLY clips of the gold train split, test negatives ONLY clips of the gold test
  split, so speakers never cross splits. Noise files are split by filename hash (80% train / 20% test).
* Every row: command=OUT_OF_SCOPE, out_of_scope=1, bucket=OUT_OF_SCOPE, source=synthetic_negative,
  is_synthetic=1, accent_group=Synthetic, transcript `<kind>`, speaker_id `neg_<kind>_<first source speaker>`.
  Extra columns: `neg_kind`, `source_files` (";"-joined, "<split>/<file>" or "noise/<name>").

| neg_kind | what it is |
|---|---|
| noise_only | a segment of a noise file at -45..-15 dBFS RMS |
| babble | 3-6 clips of the split summed at random offsets (different speakers), -30..-20 dBFS RMS |
| reversed | one gold clip played backwards |
| truncated | first 25-40% of the speech of an in-scope clip with >= 3 words and >= 1.5 s (never a full command), optionally with light noise |
| near_silence | -60..-50 dBFS noise, or digital near-silence with ~0.4-2 LSB mic hiss |

Clips are 1-5 s. Counts: {counts}

Pack with `python scripts/pack_data.py --negatives <this folder>` (splits `neg_train`, `neg_test`),
then train with `--use-negatives` and score `neg_test` with `python -m vcm.eval --split neg_test`.
"""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="dataset root with train/ and test/")
    ap.add_argument("--noise", required=True, help="folder of noise wavs")
    ap.add_argument("--out", default=None, help="default: <dataset>/synthetic_negatives")
    ap.add_argument("--n-train", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing <out>/train and <out>/test")
    a = ap.parse_args(argv)
    root = Path(a.dataset).expanduser()
    out = Path(a.out).expanduser() if a.out else root / "synthetic_negatives"
    noise_files = sorted(Path(a.noise).expanduser().glob("*.wav"))
    if not noise_files:
        raise SystemExit(f"no .wav files in {a.noise}")
    for s in ("train", "test"):
        if (out / s).exists():
            if not a.overwrite:
                raise SystemExit(f"{out / s} exists (use --overwrite)")
            shutil.rmtree(out / s)
    cols = []
    for s in ("train", "test"):
        for c in pd.read_csv(root / s / "manifest.csv", nrows=0).columns:
            if c not in cols:
                cols.append(c)
    cols += [c for c in REQUIRED_COLS if c not in cols] + [c for c in EXTRA_COLS if c not in cols]
    noise_tr, noise_te = noise_split(noise_files)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, noise, n in (("train", noise_tr, a.n_train), ("test", noise_te, a.n_test)):
        df = make_split(root, split, noise, out, n, a.seed, cols)
        counts[split] = {"total": len(df), **{k: int((df["neg_kind"] == k).sum()) for k in KINDS}}
        print(f"{split}: {counts[split]}")
    summary = {"seed": a.seed, "dataset": str(root), "noise": str(Path(a.noise).expanduser()), "counts": counts,
               "noise_files": {"train": [f.name for f in noise_tr], "test": [f.name for f in noise_te]}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    cs = "; ".join(f"{s}: " + ", ".join(f"{k} {v}" for k, v in c.items()) for s, c in counts.items())
    (out / "README.md").write_text(README.format(seed=a.seed, counts=cs))
    return summary


if __name__ == "__main__":
    main()
