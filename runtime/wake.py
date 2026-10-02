"""Sliding-window wake-word detector (pure logic; the scorer is injected so it is testable without ONNX)."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

SR = 16000


class RingBuffer:
    """last `capacity` samples of a mono float32 stream"""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf = np.zeros(capacity, np.float32)
        self.filled = 0

    def append(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        n = len(x)
        if n >= self.capacity:
            self._buf[:] = x[-self.capacity:]
        else:
            self._buf = np.roll(self._buf, -n)
            self._buf[-n:] = x
        self.filled = min(self.capacity, self.filled + n)

    def last(self, n: int) -> np.ndarray:
        n = min(n, self.filled)
        return self._buf[self.capacity - n:].copy() if n else np.zeros(0, np.float32)

    def clear(self) -> None:
        self._buf[:] = 0
        self.filled = 0


@dataclass
class WakeEvent:
    prob: float
    infer_ms: float
    t_done: float          # time.perf_counter() when the scorer returned


class WakeDetector:
    """Scores the last `window` samples every `hop` samples; fires when P(WATSON) >= threshold for
    `consecutive` scored windows in a row. After a fire (or `hold()`), scoring is suppressed until
    `cooldown_s` after `release()`."""

    def __init__(self, score_fn, window: int = 24000, hop_s: float = 0.25, threshold: float = 0.55,
                 consecutive: int = 1, cooldown_s: float = 1.0, ring: RingBuffer | None = None):
        self.score_fn = score_fn            # (audio float32) -> (prob, infer_ms)
        self.window = window
        self.hop = int(hop_s * SR)
        self.threshold = threshold
        self.consecutive = max(1, consecutive)
        self.cooldown = int(cooldown_s * SR)
        self.ring = ring or RingBuffer(max(window, SR * 3))
        self._since = 0
        self._streak = 0
        self._cool_left = 0
        self._held = False
        self.last_prob = 0.0

    def feed(self, block: np.ndarray) -> WakeEvent | None:
        block = np.asarray(block, dtype=np.float32)
        self.ring.append(block)
        if self._held:
            return None
        if self._cool_left > 0:
            self._cool_left = max(0, self._cool_left - len(block))
            self._since = 0
            return None
        self._since += len(block)
        if self.ring.filled < self.hop or self._since < self.hop:
            return None
        self._since = 0
        prob, ms = self.score_fn(self.ring.last(self.window))
        self.last_prob = prob
        self._streak = self._streak + 1 if prob >= self.threshold else 0
        if self._streak >= self.consecutive:
            self._streak = 0
            self._held = True
            return WakeEvent(prob, ms, time.perf_counter())
        return None

    def hold(self) -> None:
        """stop scoring (a command turn is in progress)"""
        self._held = True

    def release(self) -> None:
        """resume scoring after a turn; the ring is cleared so the command audio can't re-trigger the wake word"""
        self._held = False
        self._streak = 0
        self._since = 0
        self._cool_left = self.cooldown
        self.ring.clear()
