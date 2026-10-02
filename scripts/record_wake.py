"""Guided recorder for the "Watson" wake word (positives) and look-alike speech (negatives).

Each block records continuously; you say one item, pause about a second, say the next.
Afterwards the block is split into utterances by energy and each one is saved as its own clip:
  data/wake_rec/<session>/WATSON/<block>_<n>.wav
  data/wake_rec/<session>/OTHER/<block>_<n>.wav
  data/wake_rec/<session>/raw/<block>.wav          (the whole block, for re-segmenting)
  data/wake_rec/<session>/NOISE/<block>.wav        (room tone, no speech)
A session is the unit the wake splits use (whole sessions are held out), so record each
session in one sitting and use a new --session name for a new place or microphone.

  .venv/bin/python scripts/record_wake.py --session mac1              # the full guided run (~8 min)
  .venv/bin/python scripts/record_wake.py --session mac1 --only far   # redo one block
  .venv/bin/python scripts/record_wake.py --list-devices
"""
import argparse
import os
import sys
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

SR = 16000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (name, label, seconds, instruction)
BLOCKS = [
    ("normal", "WATSON", 35, "Say 'Watson' normally, about arm's length from the laptop. One every ~2 s."),
    ("quick", "WATSON", 30, "Say 'Watson' QUICK and clipped ('Wats'n'), like you're in a hurry."),
    ("slow", "WATSON", 30, "Say 'Watson' SLOW and drawn out ('Waaatsooon')."),
    ("soft", "WATSON", 30, "Say 'Watson' SOFTLY, almost under your breath (not a whisper)."),
    ("loud", "WATSON", 30, "Say 'Watson' LOUD, like calling it from across the room."),
    ("far", "WATSON", 35, "Move ~2 m away (or turn away from the laptop). Say 'Watson' normally."),
    ("tones", "WATSON", 35, "Vary it: question 'Watson?', tired, smiling, annoyed, flat, Filipino-accented."),
    ("lead", "WATSON", 40, "Say 'Watson' then a short command each time: 'Watson, lights on' ... vary the command."),
    ("confusable", "OTHER", 60, "Read these, one every ~2 s:\n"
     "  what's on | watch it | wattage | Wilson | Watsonville | Jackson | Hudson | Madison\n"
     "  what's up | not some | lots on | was it | Austin | Watts | Boston | Watching\n"
     "  hot son | what son | Dawson | Lawson | Wesson | Wasson | watts on | wash on"),
    ("talk", "OTHER", 60, "Talk normally about anything (no 'Watson'): your day, read a page, give commands without the wake word."),
    ("room", "NOISE", 20, "Stay SILENT. Room tone only."),
]


def frames_db(x, n=320):
    k = len(x) // n
    f = x[:k * n].reshape(k, n)
    return 10 * np.log10((f ** 2).mean(1) + 1e-10)


def segment(x, label, margin_db=12.0, merge_s=0.30, pad_s=0.15):
    """Return [(start, end)] sample ranges of utterances in a continuous block."""
    db = frames_db(x)
    floor = np.percentile(db, 15)
    on = db > max(floor + margin_db, -60.0)
    hop = 320
    segs, start = [], None
    for i, v in enumerate(on):
        if v and start is None:
            start = i
        elif not v and start is not None:
            segs.append([start, i])
            start = None
    if start is not None:
        segs.append([start, len(on)])
    merged = []
    gap = int(merge_s * SR / hop)
    for s in segs:
        if merged and s[0] - merged[-1][1] <= gap:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    lo, hi = (0.20, 1.8) if label == "WATSON" else (0.20, 4.0)
    if label == "WATSON":
        # 'lead' blocks run into the command; keep wake + command, the trainer crops windows
        hi = 4.0
    pad = int(pad_s * SR)
    out = []
    for a, b in merged:
        dur = (b - a) * hop / SR
        if lo <= dur <= hi:
            out.append((max(0, a * hop - pad), min(len(x), b * hop + pad)))
    return out


def record(seconds, device):
    print(f"  recording {seconds} s ... ", end="", flush=True)
    x = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32", device=device)
    t0 = time.time()
    while time.time() - t0 < seconds:
        left = seconds - (time.time() - t0)
        print(f"\r  recording ... {left:4.0f} s left ", end="", flush=True)
        time.sleep(0.5)
    sd.wait()
    print("\r  done.                       ")
    return x[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", help="name for this sitting, e.g. mac1")
    ap.add_argument("--only", nargs="+", help="block names to record (default: all)")
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--seconds-scale", type=float, default=1.0)
    ap.add_argument("--list-devices", action="store_true")
    a = ap.parse_args()
    if a.list_devices:
        print(sd.query_devices())
        return
    if not a.session:
        sys.exit("--session is required")
    base = os.path.join(ROOT, "data", "wake_rec", a.session)
    blocks = [b for b in BLOCKS if not a.only or b[0] in a.only]
    print(f"Mic: {sd.query_devices(a.device, 'input')['name']}  ->  {base}")
    totals = {}
    for name, label, secs, instr in blocks:
        secs = int(secs * a.seconds_scale)
        print(f"\n[{name}] {label}\n  {instr}")
        while True:
            input("  press Enter to start ")
            x = record(secs, a.device)
            peak = float(np.abs(x).max())
            if peak > 0.99:
                print("  clipped: back off a little")
            os.makedirs(os.path.join(base, "raw"), exist_ok=True)
            sf.write(os.path.join(base, "raw", f"{name}.wav"), x, SR, subtype="PCM_16")
            if label == "NOISE":
                os.makedirs(os.path.join(base, "NOISE"), exist_ok=True)
                sf.write(os.path.join(base, "NOISE", f"{name}.wav"), x, SR, subtype="PCM_16")
                n = 1
            else:
                segs = segment(x, label)
                d = os.path.join(base, label)
                os.makedirs(d, exist_ok=True)
                for f in os.listdir(d):
                    if f.startswith(name + "_"):
                        os.remove(os.path.join(d, f))
                for i, (s, e) in enumerate(segs):
                    sf.write(os.path.join(d, f"{name}_{i:03d}.wav"), x[s:e], SR, subtype="PCM_16")
                n = len(segs)
            print(f"  peak {peak:.2f}, {n} clip(s) saved")
            if input("  keep? [Y/n] ").strip().lower() != "n":
                totals[name] = n
                break
    print("\nsaved:", totals)


if __name__ == "__main__":
    main()
