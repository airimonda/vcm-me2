#!/usr/bin/env python3
"""Render the spoken replies in config/replies.json with Piper into replies/ (22.05 kHz mono, MP3 if ffmpeg
is on PATH else WAV, peak-normalised). The runtime plays them by key (runtime/feedback.py); a missing clip
falls back to espeak-ng, so rendering is optional but sounds much better.

Voice: Piper en_US-amy-medium + the "ship" effect (a generic sci-fi computer colouring: light ring
modulation, a metallic comb, a short room echo), length_scale 1.1, noise_scale 0.4 -- the defaults below.

    python scripts/gen_reply_audio.py --list                    # how many clips, no rendering
    python scripts/gen_reply_audio.py --limit 10                # a random 10-clip check
    python scripts/gen_reply_audio.py --limit 0                 # everything (skips existing files)
    python scripts/gen_reply_audio.py --samples                 # audition lines -> replies/_samples/
    python scripts/gen_reply_audio.py --voice-dir ~/piper_voices --voice en_US-amy-medium

Needs `pip install piper-tts soundfile numpy` (+ ffmpeg for MP3) and the voice's .onnx/.onnx.json in --voice-dir
(default data/piper_voices/). Run it on the Mac and sync replies/ to the Pi (scripts/sync_to_pi.sh).

Voice character knobs (output side only; nothing here touches the models):
  --length-scale X  >1 slower   --noise-scale X  lower = flatter intonation
  --fx none|ship|robot|idol   idol needs ffmpeg
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import shutil
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from runtime.feedback import clip_basename, render_text  # noqa: E402

CATALOGUE = os.path.join(REPO_ROOT, "config", "replies.json")
OUT_DIR = os.path.join(REPO_ROOT, "replies")
VOICE_DIR = os.path.join(REPO_ROOT, "data", "piper_voices")


# ------------------------------------------------------------------ catalogue expansion
def domain_values(slot: dict) -> list:
    if "values" in slot:
        return list(slot["values"])
    kind, *args = slot["domain"].split(":")
    if kind == "range":
        return list(range(int(args[0]), int(args[1]) + 1))
    raise ValueError(f"unknown slot domain {slot['domain']!r}")


def expand(catalogue: dict) -> list[dict]:
    """-> [{file (no extension), key, values, text}] for every (key, slot combination)"""
    rows = []
    for key, tpl in catalogue.items():
        if key.startswith("_"):
            continue
        slots = tpl.get("slots", [])
        names = [s["name"] for s in slots]
        for combo in itertools.product(*[domain_values(s) for s in slots]) if slots else [()]:
            values = dict(zip(names, combo))
            rows.append({"file": clip_basename(key, tpl, values), "key": key, "values": values,
                         "text": render_text(tpl, values)})
    return rows


# ------------------------------------------------------------------ audio
def _single_thread_onnx():
    """one onnxruntime thread per process: Piper's default grabs every core and stalls the Mac"""
    import onnxruntime
    if getattr(onnxruntime, "_vcm_patched", False):
        return
    base = onnxruntime.SessionOptions

    def one_thread():
        o = base()
        o.intra_op_num_threads = 1
        o.inter_op_num_threads = 1
        return o
    onnxruntime.SessionOptions = one_thread
    onnxruntime._vcm_patched = True


def load_voice(voice_dir, name):
    _single_thread_onnx()
    from piper import PiperVoice
    return PiperVoice.load(os.path.join(voice_dir, name + ".onnx"))


SAMPLE_LINES = [
    "Timer set for 30 seconds.",
    "Lights blue.",
    "It's 27 degrees and partly cloudy in Quezon City.",
    "Calling Mom.",
    "Sorry, I didn't catch that.",
]

# post-effects: ring = (mix, carrier Hz); comb = (delay ms, gain); echo = (delay ms, gain)
FX = {
    "none": None,
    "ship": {"ring": (0.25, 40.0), "comb": (4.0, 0.35), "echo": (70.0, 0.15), "highpass": 180.0},
    "robot": {"ring": (0.6, 55.0), "comb": (3.0, 0.5), "echo": (50.0, 0.2), "highpass": 220.0},
    "idol": {},          # done in ffmpeg at encode time (FFMPEG_FX)
}
FFMPEG_FX = {
    "idol": "asetrate={sr}*1.26,aresample={sr},atempo=0.84,"
            "chorus=0.7:0.9:40:0.25:0.3:1.6,treble=g=5,highpass=f=150,alimiter=limit=0.9",
}


def apply_fx(wav, sr, preset):
    """sci-fi computer colouring in numpy: ring modulation (metallic buzz), a short comb (tinny resonance), a
    light echo (a room/console), a first-order high-pass (thin speaker); re-peak-normalised afterwards."""
    import numpy as np
    fx = FX.get(preset)
    if not fx or "ring" not in fx:
        return wav
    t = np.arange(len(wav)) / sr
    mix, hz = fx["ring"]
    out = wav * ((1 - mix) + mix * np.sin(2 * np.pi * hz * t))
    for k in ("comb", "echo"):
        ms, gain = fx[k]
        d = int(sr * ms / 1000)
        delayed = np.zeros_like(out)
        delayed[d:] = out[:-d] * gain
        out = out + delayed
    rc = 1.0 / (2 * np.pi * fx["highpass"])
    alpha = rc / (rc + 1.0 / sr)
    hp = np.zeros_like(out)
    for i in range(1, len(out)):          # clips are ~1-3 s, a plain loop is fine
        hp[i] = alpha * (hp[i - 1] + out[i] - out[i - 1])
    out = np.concatenate([hp, np.zeros(int(sr * fx["echo"][0] / 1000))])     # let the echo ring out
    peak = np.abs(out).max()
    return np.clip(out / peak * 0.9, -1, 1) if peak > 0 else out


def synth_text(voice, text, speaker=None, length_scale=None, noise_scale=None):
    import numpy as np
    from piper import SynthesisConfig
    cfg = SynthesisConfig()
    if speaker is not None:
        cfg.speaker_id = speaker
    if length_scale is not None:
        cfg.length_scale = length_scale
    if noise_scale is not None:
        cfg.noise_scale = noise_scale
    audio = []
    for sent in voice.phonemize(text):
        audio.append(voice.phoneme_ids_to_audio(voice.phonemes_to_ids(sent), cfg))
    wav = np.concatenate(audio).astype(np.float32)
    if np.abs(wav).max() > 1.5:                     # int16-scaled output
        wav = wav / 32768.0
    peak = np.abs(wav).max()
    if peak > 0:
        wav = wav / peak * 0.9
    return np.clip(wav, -1, 1)


def encode(wav, sr, out_path, use_ffmpeg, fx="none"):
    import soundfile as sf
    if not use_ffmpeg:
        sf.write(out_path, wav, sr, subtype="PCM_16")
        return
    tmp = out_path + ".tmp.wav"
    sf.write(tmp, wav, sr, subtype="PCM_16")
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
                        *(["-af", FFMPEG_FX[fx].format(sr=sr)] if fx in FFMPEG_FX else []),
                        "-ac", "1", "-c:a", "libmp3lame", "-b:a", "48k", out_path], check=True)
    finally:
        os.remove(tmp)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--catalogue", default=CATALOGUE)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--voice-dir", default=VOICE_DIR, help="folder with <voice>.onnx and <voice>.onnx.json")
    ap.add_argument("--voice", default="en_US-amy-medium")
    ap.add_argument("--limit", type=int, default=10, help="clips to render this run; 0 = all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speaker", type=int, default=None)
    ap.add_argument("--length-scale", type=float, default=1.1)
    ap.add_argument("--noise-scale", type=float, default=0.4)
    ap.add_argument("--fx", default="ship", choices=sorted(FX))
    ap.add_argument("--samples", action="store_true", help="render SAMPLE_LINES to <out>/_samples/ for auditioning")
    ap.add_argument("--list", action="store_true", help="print the number of clips and exit")
    a = ap.parse_args()
    synth = dict(speaker=a.speaker, length_scale=a.length_scale, noise_scale=a.noise_scale)

    rows = expand(json.load(open(a.catalogue)))
    if a.list:
        per_key = {}
        for r in rows:
            per_key[r["key"]] = per_key.get(r["key"], 0) + 1
        print(f"{len(rows)} clips from {len(per_key)} keys")
        for k, n in per_key.items():
            print(f"  {k}: {n}")
        return

    import numpy as np
    use_ffmpeg = shutil.which("ffmpeg") is not None
    voice = load_voice(os.path.expanduser(a.voice_dir), a.voice)
    sr = voice.config.sample_rate
    if a.samples:
        gap = np.zeros(int(sr * 0.6), dtype=np.float32)
        parts = []
        for line in SAMPLE_LINES:
            parts += [apply_fx(synth_text(voice, line, **synth), sr, a.fx), gap]
        d = os.path.join(a.out, "_samples")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{a.voice}_{a.fx}" + (".mp3" if use_ffmpeg else ".wav"))
        encode(np.concatenate(parts), sr, path, use_ffmpeg, a.fx)
        print(path)
        return

    if a.limit:
        random.Random(a.seed).shuffle(rows)
        rows = rows[:a.limit]
    os.makedirs(a.out, exist_ok=True)
    rendered = skipped = 0
    for row in rows:
        path = os.path.join(a.out, row["file"] + (".mp3" if use_ffmpeg else ".wav"))
        if os.path.exists(path):
            skipped += 1
            continue
        wav = apply_fx(synth_text(voice, row["text"], **synth), sr, a.fx)
        encode(wav, sr, path, use_ffmpeg, a.fx)
        rendered += 1
        if rendered % 50 == 0:
            print(f"  {rendered} rendered...", flush=True)
    print(f"rendered {rendered}, skipped (existing) {skipped}; voice={a.voice} fx={a.fx} "
          f"format={'mp3' if use_ffmpeg else 'wav'} -> {a.out}")


if __name__ == "__main__":
    main()
