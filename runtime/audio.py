"""Audio input sources. Every source is an async iterator of float32 mono 16 kHz blocks.

* SoundDeviceSource : PortAudio via sounddevice, default device (PipeWire echo-cancel source on the Pi).
* ArecordSource     : `arecord` subprocess fallback when sounddevice/PortAudio is unavailable.
* WavSource         : feeds a WAV file (testing, --input-wav); ends when the file ends.
* InjectSource      : wraps the mic; audio posted to the dashboard's /inject endpoint replaces the mic
                      signal block by block, in step with the mic clock (benchmark without a loudspeaker).

Blocks are `block_ms` long. The callback thread hands them to the asyncio loop with call_soon_threadsafe.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import subprocess
import threading
from math import gcd

import numpy as np

SR = 16000
log = logging.getLogger("audio")


def load_wav(path, sr: int = SR) -> np.ndarray:
    """mono float32 at 16 kHz from any WAV soundfile can read (polyphase resample if needed)"""
    import soundfile as sf
    x, rate = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if rate != sr:
        try:
            from scipy.signal import resample_poly
            g = gcd(sr, rate)
            x = resample_poly(x, sr // g, rate // g).astype(np.float32)
        except ImportError:                          # the Pi venv has no scipy: linear interpolation
            n = int(round(len(x) * sr / rate))
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)
    return x.astype(np.float32)


class Source:
    block: int

    def __aiter__(self):
        return self

    async def __anext__(self) -> np.ndarray:           # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        pass

    def flush(self) -> None:
        """drop audio queued while the loop was busy (no-op for files)"""


class WavSource(Source):
    """Streams `audio` in blocks, followed by `tail_s` of silence (so the endpointer sees the end of
    speech). With realtime=True blocks are paced at wall-clock speed (dashboard demos); otherwise as
    fast as the consumer pulls them."""

    def __init__(self, audio: np.ndarray, block_ms: int = 100, lead_s: float = 0.0, tail_s: float = 1.5,
                 realtime: bool = False):
        self.block = int(SR * block_ms / 1000)
        self.audio = np.concatenate([np.zeros(int(lead_s * SR), np.float32), np.asarray(audio, np.float32),
                                     np.zeros(int(tail_s * SR), np.float32)])
        self.pos = 0
        self.realtime = realtime
        self.block_s = block_ms / 1000

    async def __anext__(self):
        if self.pos >= len(self.audio):
            raise StopAsyncIteration
        blk = self.audio[self.pos: self.pos + self.block]
        self.pos += self.block
        if len(blk) < self.block:
            blk = np.pad(blk, (0, self.block - len(blk)))
        await asyncio.sleep(self.block_s if self.realtime else 0)
        return blk

    @classmethod
    def from_file(cls, path, **kw):
        return cls(load_wav(path), **kw)


class _QueueSource(Source):
    def __init__(self, block_ms: int, queue_max: int = 200):
        self.block = int(SR * block_ms / 1000)
        self._loop = asyncio.get_running_loop()
        self._q: asyncio.Queue = asyncio.Queue(maxsize=queue_max)
        self.dropped = 0

    def _put(self, blk):                              # runs on the event loop
        try:
            self._q.put_nowait(blk)
        except asyncio.QueueFull:                     # consumer stalled: drop the oldest, keep streaming
            self.dropped += 1
            try:
                self._q.get_nowait()
                self._q.put_nowait(blk)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass

    def _threadsafe_put(self, blk):
        try:
            self._loop.call_soon_threadsafe(self._put, blk)
        except RuntimeError:                          # loop closed during shutdown
            pass

    async def __anext__(self):
        return await self._q.get()

    def flush(self) -> int:
        n = 0
        while not self._q.empty():
            self._q.get_nowait()
            n += 1
        return n


class SoundDeviceSource(_QueueSource):
    def __init__(self, device=None, block_ms: int = 100):
        super().__init__(block_ms)
        import sounddevice as sd
        self.overflows = 0
        self._stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=self.block,
                                      device=device, callback=self._cb)
        self._stream.start()

    def _cb(self, indata, frames, time_info, status):
        if status:
            self.overflows += 1
        self._threadsafe_put(indata[:, 0].copy())

    def close(self):
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass


class ArecordSource(_QueueSource):
    """arecord -> raw s16le on stdout, read by a thread"""

    def __init__(self, device=None, block_ms: int = 100):
        if not shutil.which("arecord"):
            raise RuntimeError("arecord not found")
        super().__init__(block_ms)
        argv = ["arecord", "-q", "-f", "S16_LE", "-r", str(SR), "-c", "1", "-t", "raw"]
        if device:
            argv += ["-D", str(device)]
        self._proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        nbytes = self.block * 2
        while True:
            raw = self._proc.stdout.read(nbytes)
            if len(raw) < nbytes:
                break
            self._threadsafe_put(np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0)

    def close(self):
        self._proc.terminate()


class InjectSource(Source):
    """The mic, except while injected audio is pending: then each mic block is replaced by the next
    block of the injected clip. The clip advances with the mic's own clock, also over blocks dropped by
    flush(), so it plays out in real time just as sound from a loudspeaker would."""

    def __init__(self, inner: Source):
        self.inner = inner
        self.block = inner.block
        self._clip = None
        self._pos = 0
        self.injected = 0

    def inject(self, audio: np.ndarray) -> float:
        """queue a clip (replaces any clip still playing); returns its length in seconds"""
        self._clip = np.asarray(audio, np.float32).reshape(-1)
        self._pos = 0
        self.injected += 1
        return len(self._clip) / SR

    @property
    def active(self) -> bool:
        return self._clip is not None

    def _advance(self, n: int) -> np.ndarray | None:
        if self._clip is None:
            return None
        blk = self._clip[self._pos: self._pos + n]
        self._pos += n
        if self._pos >= len(self._clip):
            self._clip = None
        return np.pad(blk, (0, n - len(blk))) if len(blk) < n else blk

    async def __anext__(self):
        blk = await self.inner.__anext__()
        rep = self._advance(len(blk))
        return blk if rep is None else rep

    def flush(self) -> int:
        n = self.inner.flush() or 0
        if n:
            self._advance(n * self.block)
        return n

    def close(self) -> None:
        self.inner.close()


def open_mic(backend: str = "auto", device=None, block_ms: int = 100) -> Source:
    """microphone source; `auto` tries sounddevice, then arecord"""
    if backend in ("auto", "sounddevice"):
        try:
            return SoundDeviceSource(device, block_ms)
        except Exception as e:
            if backend == "sounddevice":
                raise
            log.warning("sounddevice unavailable (%s); falling back to arecord", e)
    return ArecordSource(device, block_ms)
