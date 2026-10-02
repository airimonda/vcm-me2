"""Energy-based endpointing for the command capture that follows the wake word.

Feed it 16 kHz float32 audio in any chunk size. It works on 20 ms frames:

* the speech threshold is `noise_db + margin_db` (never below `min_db`); `noise_db` is the noise floor
  measured on the audio before the wake word (see `noise_floor_db`);
* speech STARTS after `min_speech_s` of consecutive frames above the threshold, but frames in the first
  `start_ignore_s` are not allowed to start it (the earcon / the tail of "Watson"); the audio itself is
  always kept, so nothing is cut off;
* speech ENDS `end_silence_s` after the last voiced frame (a frame is voiced if above the start threshold
  AND within 35 dB of the loudest frame so far), but not before `min_s` of audio exists;
* hard stop at `max_s`; if no speech starts within `no_speech_timeout_s` the capture ends with reason
  "no_speech".
"""
from __future__ import annotations

import time

import numpy as np

SR = 16000
FRAME = 320                      # 20 ms
REL_DB = 35.0                    # voiced = within this many dB of the loudest frame (matches prep.trim_speech)


def frame_db(x: np.ndarray) -> np.ndarray:
    """dBFS per 20 ms frame (len(x) // 320 frames)"""
    n = len(x) // FRAME
    if n == 0:
        return np.zeros(0)
    f = np.asarray(x[: n * FRAME], dtype=np.float64).reshape(n, FRAME)
    return 10.0 * np.log10((f ** 2).mean(axis=1) + 1e-12)


def noise_floor_db(x: np.ndarray, default: float = -60.0) -> float:
    """noise floor of a pre-wake audio snippet: 10th percentile of its frame energies"""
    db = frame_db(x)
    return float(np.percentile(db, 10)) if len(db) >= 5 else default


class Endpointer:
    def __init__(self, noise_db: float = -60.0, max_s: float = 5.0, min_s: float = 0.8,
                 end_silence_s: float = 0.7, no_speech_timeout_s: float = 3.0, start_ignore_s: float = 0.25,
                 margin_db: float = 10.0, min_db: float = -55.0, min_speech_s: float = 0.12):
        self.thr_db = max(noise_db + margin_db, min_db)
        self.max_n = int(max_s * SR)
        self.min_n = int(min_s * SR)
        self.end_frames = max(1, round(end_silence_s / (FRAME / SR)))
        self.timeout_frames = round(no_speech_timeout_s / (FRAME / SR))
        self.ignore_frames = round(start_ignore_s / (FRAME / SR))
        self.start_frames = max(1, round(min_speech_s / (FRAME / SR)))
        self._chunks: list[np.ndarray] = []
        self._pending = np.zeros(0, np.float32)
        self.n_samples = 0
        self.n_frames = 0
        self.speech_started = False
        self.speech_start_frame: int | None = None
        self.last_voiced_frame: int | None = None
        self.last_voiced_wall: float | None = None   # time.perf_counter() when the last voiced frame was processed
        self._run = 0
        self._peak_db = -120.0
        self.done = False
        self.reason = ""

    # ------------------------------------------------------------------
    def feed(self, block: np.ndarray) -> bool:
        """Add audio; returns True once the capture is finished (see .reason)."""
        if self.done:
            return True
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        self._chunks.append(block)
        self.n_samples += len(block)
        buf = np.concatenate([self._pending, block])
        nfr = len(buf) // FRAME
        self._pending = buf[nfr * FRAME:]
        for db in frame_db(buf):
            self._frame(float(db))
            if self.done:
                break
        if not self.done and self.n_samples >= self.max_n:
            self._finish("max_length")
        return self.done

    def finish_eof(self) -> None:
        """the audio source ended (wav replay): close whatever is open"""
        if not self.done:
            self._finish("source_ended" if self.speech_started else "no_speech")

    def _frame(self, db: float) -> None:
        i = self.n_frames
        self.n_frames += 1
        self._peak_db = max(self._peak_db, db)
        loud = db >= self.thr_db
        if not self.speech_started:
            if i >= self.ignore_frames and loud:
                self._run += 1
                if self._run >= self.start_frames:
                    self.speech_started = True
                    self.speech_start_frame = i - self._run + 1
                    self.last_voiced_frame = i
                    self.last_voiced_wall = time.perf_counter()
            else:
                self._run = 0
            if not self.speech_started and i + 1 >= self.timeout_frames:
                self._finish("no_speech")
            return
        if loud and db >= self._peak_db - REL_DB:
            self.last_voiced_frame = i
            self.last_voiced_wall = time.perf_counter()
        silent_for = i - self.last_voiced_frame
        if silent_for >= self.end_frames and self.n_samples >= self.min_n:
            self._finish("end_of_speech")

    def _finish(self, reason: str) -> None:
        self.done, self.reason = True, reason

    # ------------------------------------------------------------------
    @property
    def audio(self) -> np.ndarray:
        return np.concatenate(self._chunks) if self._chunks else np.zeros(0, np.float32)

    @property
    def has_speech(self) -> bool:
        return self.speech_started

    @property
    def speech_end_s(self) -> float | None:
        """time (s since capture start) the last voiced frame ended"""
        return None if self.last_voiced_frame is None else (self.last_voiced_frame + 1) * FRAME / SR
