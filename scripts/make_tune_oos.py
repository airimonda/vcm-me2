"""Build an EVALUATION-ONLY out-of-scope set for the tune split, so the reject threshold (tau) is
set on hundreds of out-of-scope clips instead of the ~28 gold ones in tune.

Clips come from the collated pool (~/ai231-me2-collated/pool.csv), are OUT_OF_SCOPE, and belong to
speakers that are in none of train, test or holdout (so the set is unseen by training and says
nothing about test). Mix, following the gold out-of-scope recipe minus noise (neg_tune covers noise):
near-miss requests 50% (SLURP / FSC / SNIPS / TimersAndSuch), Filipino-accented speech up to 25%
(Common Voice, accent_group "Filipino (open-source)"), general speech the rest (Common Voice).

Writes <out>/tune_oos/manifest.csv + audio/*.wav (16 kHz mono 16-bit), the layout pack_data.py reads:
  python scripts/make_tune_oos.py --collated ~/ai231-me2-collated --out data/tune_oos_src --n 400
  python scripts/pack_data.py --dataset data/tune_oos_src --splits tune_oos --out data/packs
Never used for training.
"""
import argparse
import os

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

NEAR_MISS = ["SLURP", "FluentSpeechCommands", "SNIPS", "TimersAndSuch"]


def load_16k(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype("float32")
    return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collated", default="~/ai231-me2-collated")
    ap.add_argument("--out", default="data/tune_oos_src")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    root = os.path.expanduser(a.collated)
    pool = pd.read_csv(os.path.join(root, "pool.csv"), dtype=str, keep_default_na=False)
    used = set()
    cols = None
    for s in ("train", "test", "holdout"):
        m = pd.read_csv(os.path.join(root, "dataset", s, "manifest.csv"), dtype=str, keep_default_na=False)
        used |= set(m.speaker_id)
        cols = cols or [c for c in m.columns if c != "original_path"]
    o = pool[(pool.command == "OUT_OF_SCOPE") & (~pool.speaker_id.isin(used)) & (pool.speaker_id != "")]
    o = o[o.audio_path.map(os.path.exists)]
    rng = np.random.RandomState(a.seed)

    def take(df, n):
        # at most 2 clips per speaker, so many voices are represented
        df = df.sample(frac=1, random_state=rng).groupby("speaker_id").head(2)
        return df.head(n)

    fil = take(o[(o.accent_group == "Filipino (open-source)") & (o.source == "CommonVoice_en")], a.n // 4)
    near = take(o[o.source.isin(NEAR_MISS)], a.n // 2)
    gen = take(o[(o.source == "CommonVoice_en") & (o.accent_group != "Filipino (open-source)")],
               a.n - len(fil) - len(near))
    sel = pd.concat([near.assign(kind="near_miss"), fil.assign(kind="filipino"), gen.assign(kind="general")])

    out = os.path.join(a.out, "tune_oos")
    os.makedirs(os.path.join(out, "audio"), exist_ok=True)
    rows = []
    for i, r in enumerate(sel.itertuples()):
        rel = f"audio/tune_oos_{i:05d}.wav"
        x = load_16k(r.audio_path)
        sf.write(os.path.join(out, rel), x, 16000, subtype="PCM_16")
        rows.append({"file": rel, "transcript": r.transcript, "command": "OUT_OF_SCOPE", "variation": "",
                     "slot_value": "", "out_of_scope": "1", "bucket": "OUT_OF_SCOPE", "speaker_id": r.speaker_id,
                     "source": r.source, "is_synthetic": "0", "accent_group": r.accent_group,
                     "numerals": r.numerals, "duration_s": f"{len(x) / 16000:.3f}",
                     "transcript_source": r.transcript_source, "oos_kind": r.kind})
    m = pd.DataFrame(rows)
    for c in cols:
        if c not in m:
            m[c] = ""
    m.to_csv(os.path.join(out, "manifest.csv"), index=False)
    print(len(m), "clips;", m.oos_kind.value_counts().to_dict(), "speakers", m.speaker_id.nunique())


if __name__ == "__main__":
    main()
