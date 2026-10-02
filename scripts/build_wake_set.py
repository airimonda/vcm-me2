"""Collect the wake-word clips into two packs for scripts/train_wake.py.

  data/wake/train.npz, data/wake/tune.npz with
    audio   int16, all clips concatenated (16 kHz mono, clips cut to <= --max-s)
    offsets int64 (n+1), clip i = audio[offsets[i]:offsets[i+1]]
    label   int8: 1 = WATSON, 0 = OTHER (speech or anything that must not wake), 2 = NOISE (room tone)
    onset, end  int32: speech span inside the clip (energy based), for placing the wake word in the window
    lead    int8: 1 = the clip runs on into a command after "Watson" (place it at the window start)
    real    int8: 1 = a human recording, 0 = Piper
    src     str: where the clip came from (for the report)

Sources (train / tune):
  Piper (scripts/gen_piper.py --templates wake): its train / dev split (dev = held-out voices)
  your recorder sessions data/wake_rec/<session>/: train, except sessions named in --tune-sessions
  the earlier "Watson" recordings (ai231-me2-voice-data/recordings/ailene_wake: 40 WAKE + 20 NOT_WAKE)
      and the vcm-benchmark wake takes (vcm-benchmark/runs/*/wake): tune only
  the earlier project's real speech negatives (CommonVoice / SpeechCommands / SLURP / own recordings,
      from its wake manifest, files that still exist): its train / dev+own split, capped
The command dataset (packs on the training box) is added as more negatives by train_wake.py.

  .venv/bin/python scripts/build_wake_set.py --tune-sessions mac2
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import soundfile as sf

SR = 16000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOME = os.path.expanduser("~")


def load(path, max_s):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        import soxr
        x = soxr.resample(x, sr, SR)
    return x[: int(max_s * SR)]


def span(x, hop=320):
    """(onset, end) samples of the speech in x: frames above max(floor + 10 dB, peak - 35 dB)."""
    k = len(x) // hop
    if k < 2:
        return 0, len(x)
    db = 10 * np.log10((x[: k * hop].reshape(k, hop) ** 2).mean(1) + 1e-10)
    thr = max(np.percentile(db, 10) + 10, db.max() - 35)
    on = np.nonzero(db > thr)[0]
    if not len(on):
        return 0, len(x)
    return int(on[0] * hop), int(min(len(x), (on[-1] + 1) * hop))


class Pack:
    def __init__(self):
        self.rows, self.clips = [], []

    def add(self, x, label, src, real, lead=0):
        if len(x) < 1600:
            return
        a, b = span(x) if label != 2 else (0, len(x))
        peak = np.abs(x).max()
        if peak > 1e-4 and label != 2:
            x = x / peak * min(0.9, max(peak, 0.1))       # very quiet takes up to a usable level
        self.clips.append(np.clip(x * 32767, -32768, 32767).astype(np.int16))
        self.rows.append((label, a, b, lead, real, src))

    def save(self, path):
        lab, on, end, lead, real, src = zip(*self.rows)
        off = np.zeros(len(self.clips) + 1, np.int64)
        off[1:] = np.cumsum([len(c) for c in self.clips])
        np.savez(path, audio=np.concatenate(self.clips), offsets=off, label=np.array(lab, np.int8),
                 onset=np.array(on, np.int32), end=np.array(end, np.int32), lead=np.array(lead, np.int8),
                 real=np.array(real, np.int8), src=np.array(src))
        s = pd.DataFrame({"label": lab, "src": src}).groupby(["src", "label"]).size()
        print(f"{path}: {len(self.clips)} clips, {off[-1] / SR / 3600:.2f} h\n{s.to_string()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--piper", default=os.path.join(ROOT, "data/wake/piper.csv"))
    ap.add_argument("--rec", default=os.path.join(ROOT, "data/wake_rec"))
    ap.add_argument("--tune-sessions", nargs="*", default=[])
    ap.add_argument("--old-wake", default=os.path.join(HOME, "ai231-me2-voice-data/recordings/ailene_wake"))
    ap.add_argument("--bench-runs", default=os.path.join(HOME, "vcm-benchmark/runs"))
    ap.add_argument("--old-manifest", default=os.path.join(HOME, "voice-dataset-analysis/manifests/wake_labeled.csv"))
    ap.add_argument("--max-real-neg", type=int, default=12000, help="cap on the old real negatives (train)")
    ap.add_argument("--max-s", type=float, default=6.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "data/wake"))
    a = ap.parse_args()
    tr, tu = Pack(), Pack()

    p = pd.read_csv(a.piper)
    for r in p.itertuples():
        pk = tu if r.split == "dev" else tr
        lead = int(r.label == "WATSON" and "," in str(r.transcript_normalized) or
                   (r.label == "WATSON" and len(str(r.transcript_normalized).split()) > 2))
        pk.add(load(r.audio_path, a.max_s), int(r.label == "WATSON"), "piper", 0, lead)

    for sess in sorted(glob.glob(os.path.join(os.path.abspath(a.rec), "*"))):
        name = os.path.basename(sess)
        pk = tu if name in a.tune_sessions else tr
        for lab, code in (("WATSON", 1), ("OTHER", 0), ("NOISE", 2)):
            for f in sorted(glob.glob(os.path.join(sess, lab, "*.wav"))):
                lead = int(code == 1 and os.path.basename(f).startswith("lead_"))
                pk.add(load(f, 60 if code == 2 else a.max_s), code, f"rec:{name}", 1, lead)
        for blk in ("confusable", "talk"):                  # the whole block too: continuous speech to crop
            f = os.path.join(sess, "raw", blk + ".wav")
            if os.path.exists(f):
                pk.add(load(f, 120), 0, f"rec:{name}", 1)

    m = pd.read_csv(os.path.join(a.old_wake, "manifest.csv"))
    for r in m.itertuples():
        tu.add(load(os.path.join(a.old_wake, r.filename), a.max_s), int(r.label == "WAKE"), "own_wake_old", 1)
    for f in sorted(glob.glob(os.path.join(a.bench_runs, "*/wake/*.wav"))):
        tu.add(load(f, a.max_s), 1, "bench_takes", 1)

    om = pd.read_csv(a.old_manifest)
    om = om[(om.label == "OTHER")]
    om = om[om.audio_path.map(os.path.exists)]
    otr = om[om.split == "train"].sample(frac=1, random_state=0).head(a.max_real_neg)
    otu = om[om.split.isin(["dev", "own"]) & (om.dataset != "own_wake")]   # own_wake = the old_wake folder
    for d, pk in ((otr, tr), (otu, tu)):
        for r in d.itertuples():
            pk.add(load(r.audio_path, a.max_s), 0, f"real:{r.dataset}", 1)

    os.makedirs(a.out, exist_ok=True)
    tr.save(os.path.join(a.out, "train.npz"))
    tu.save(os.path.join(a.out, "tune.npz"))


if __name__ == "__main__":
    main()
