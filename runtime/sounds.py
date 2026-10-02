"""Earcons (wake / success / reject / ring) for every platform.

The Mac uses the System sounds (afplay -v) so a dev run is audible. Anywhere else (the Pi) the cues are
short synthesised WAVs written once to runtime/sounds/ and played with the first player found: pw-play,
paplay, aplay, then mpv. Nothing here blocks: `play()` starts the player and returns.

Volume: the amplitude of the synthesised tones (default 0.15 -- the value that sounded right on the
speaker without the chime leaking into the mic). There is deliberately NO guard time after the chime:
muting the mic for a moment after it cut off the first word of the command.
"""
from __future__ import annotations

import logging
import math
import os
import platform
import shutil
import struct
import subprocess
import wave

log = logging.getLogger("sounds")

SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")
SR = 22050
DEFAULT_VOLUME = 0.15
DARWIN_SOUNDS = {"wake": "Tink", "success": "Glass", "reject": "Pop", "ring": "Sosumi"}
_enabled = True

# name -> [(frequency Hz, seconds), ...]; a 0 Hz note is a rest.
NOTES = {
    "wake": [(880, 0.07), (1320, 0.09)],
    "success": [(784, 0.08), (1047, 0.08), (1319, 0.14)],
    "reject": [(330, 0.10), (247, 0.16)],
    "ring": [(1047, 0.12), (0, 0.06), (1047, 0.12), (0, 0.06), (1047, 0.12)],
}


def set_enabled(on: bool) -> None:
    global _enabled
    _enabled = bool(on)


def _synth(notes, volume):
    out = []
    for freq, dur in notes:
        n = int(SR * dur)
        for i in range(n):
            env = min(1.0, i / (SR * 0.008), (n - i) / (SR * 0.02))  # 8 ms attack, 20 ms release: no clicks
            s = math.sin(2 * math.pi * freq * i / SR) if freq else 0.0
            out.append(int(32767 * volume * env * s))
    return struct.pack("<%dh" % len(out), *out)


def ensure_wavs(volume=DEFAULT_VOLUME, directory=SOUNDS_DIR):
    """write any missing <name>_v<vol>.wav; returns {name: path}"""
    os.makedirs(directory, exist_ok=True)
    paths = {}
    for name, notes in NOTES.items():
        path = os.path.join(directory, f"{name}_v{int(round(volume * 100)):02d}.wav")
        if not os.path.exists(path):
            with wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(_synth(notes, volume))
        paths[name] = path
    return paths


def _argv(path):
    for player in ("pw-play", "paplay", "aplay"):
        if shutil.which(player):
            return [player, path] if player != "aplay" else ["aplay", "-q", path]
    if shutil.which("mpv"):
        return ["mpv", "--no-video", "--really-quiet", path]
    return None


def play(name="wake", volume=DEFAULT_VOLUME):
    """start the earcon and return immediately; False if nothing could play it"""
    if not _enabled:
        return False
    if platform.system() == "Darwin":
        argv = ["afplay", "-v", f"{volume:.2f}",
                f"/System/Library/Sounds/{DARWIN_SOUNDS.get(name, 'Tink')}.aiff"]
    else:
        paths = ensure_wavs(volume)
        argv = _argv(paths.get(name, paths["wake"]))
        if argv is None:
            log.warning("no audio player found for earcon %r", name)
            return False
    try:
        subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except OSError as e:
        log.warning("could not play earcon %r: %s", name, e)
        return False


if __name__ == "__main__":
    print(ensure_wavs())
