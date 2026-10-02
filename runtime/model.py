"""ONNX models: the command model (decision rule) and the wake-word model.

Command model: input "waveform" float32 (B, 80000) 16 kHz in [-1, 1]; outputs "cmd_logits" (B, 20)
and "slot_logits" (B, 6, 3). Labels, slot values and tau come from the JSON sidecar next to the .onnx
(never hard-coded here). Decision rule (sidecar "decision_rule"): softmax(cmd_logits); if the max
probability is below tau -> OUT_OF_SCOPE; else argmax; the slot is the argmax of that command's slot head.

Wake model: input "wav" (1, 24000) = 1.5 s, output "logits" -> labels ["WATSON", "OTHER"].
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

OOS = "OUT_OF_SCOPE"


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def make_session(path, intra_threads: int = 1):
    """ONNX Runtime CPU session tuned for the Pi: no spinning worker threads (they burn a core while idle)."""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(intra_threads)
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.log_severity_level = 3
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


@dataclass
class Decision:
    command: str                      # one of the 19 commands or OUT_OF_SCOPE (after the tau rule)
    slot: str | None                  # slot value string for slotted commands, else None
    prob: float                       # softmax prob of the top class
    slot_prob: float | None = None
    raw_command: str = ""             # argmax class before the tau rule (for logs)
    probs: list = field(default_factory=list)   # 20 class probabilities

    @property
    def is_oos(self) -> bool:
        return self.command == OOS


class Labels:
    """label space read from the sidecar"""

    def __init__(self, sidecar: dict):
        self.classes: list[str] = list(sidecar["classes"])
        self.slot_commands: list[str] = list(sidecar["slot_commands"])
        self.slot_values: dict[str, list[str]] = {k: list(v) for k, v in sidecar["slot_values"].items()}
        self.tau: float = float(sidecar.get("tau", 0.5))
        self.window: int = int(sidecar.get("window_samples", 80000))
        self.oos_idx = self.classes.index(OOS)

    @property
    def commands(self) -> list[str]:
        return [c for c in self.classes if c != OOS]


def decide(cmd_logits, slot_logits, labels: Labels, tau: float | None = None) -> Decision:
    """The decision rule on one clip. cmd_logits (20,), slot_logits (6, 3)."""
    tau = labels.tau if tau is None else float(tau)
    p = softmax(np.asarray(cmd_logits, dtype=np.float64))
    idx = int(np.argmax(p))
    prob = float(p[idx])
    raw = labels.classes[idx]
    if prob < tau or idx == labels.oos_idx:
        return Decision(OOS, None, prob, None, raw, [float(x) for x in p])
    slot, sprob = None, None
    if raw in labels.slot_commands:
        head = labels.slot_commands.index(raw)
        sp = softmax(np.asarray(slot_logits[head], dtype=np.float64))
        si = int(np.argmax(sp))
        slot, sprob = labels.slot_values[raw][si], float(sp[si])
    return Decision(raw, slot, prob, sprob, raw, [float(x) for x in p])


class CommandModel:
    def __init__(self, onnx_path, tau: float | None = None, intra_threads: int = 2):
        self.path = Path(onnx_path)
        side = self.path.with_suffix(".json")
        if not side.exists():
            raise FileNotFoundError(f"model sidecar not found: {side}")
        self.sidecar = json.loads(side.read_text())
        self.labels = Labels(self.sidecar)
        self.tau = self.labels.tau if tau is None else float(tau)
        self.sess = make_session(self.path, intra_threads)
        self.input_name = self.sess.get_inputs()[0].name
        self.window = self.labels.window

    def info(self) -> dict:
        s = self.sidecar
        return {"name": self.path.stem, "params": s.get("params"), "version": s.get("format"),
                "arch": s.get("arch"), "tau": self.tau}

    def logits(self, window: np.ndarray):
        x = np.asarray(window, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != self.window:
            raise ValueError(f"expected {self.window} samples, got {x.shape[1]}")
        cmd, slot = self.sess.run(None, {self.input_name: x})
        return cmd[0], slot[0]

    def classify(self, window: np.ndarray) -> Decision:
        cmd, slot = self.logits(window)
        return decide(cmd, slot, self.labels, self.tau)

    def warmup(self):
        self.logits(np.zeros(self.window, np.float32))


class WakeModel:
    """WATSON-vs-OTHER scorer for 1.5 s windows."""

    def __init__(self, onnx_path, intra_threads: int = 1):
        self.path = Path(onnx_path)
        # models/wake/vcm_wake_int8.onnx -> vcm_wake_int8.labels.json, else vcm_wake.labels.json, else the default
        stem = self.path.stem
        cands = [self.path.parent / (stem + ".labels.json"),
                 self.path.parent / (stem.removesuffix("_int8") + ".labels.json")]
        lab = next((c for c in cands if c.exists()), None)
        self.labels = json.loads(lab.read_text()) if lab else ["WATSON", "OTHER"]
        self.wake_idx = self.labels.index("WATSON")
        self.sess = make_session(self.path, intra_threads)
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.window = int(inp.shape[1])

    def score(self, audio: np.ndarray) -> float:
        """P(WATSON) for the last `window` samples of `audio` (zero-padded on the left if shorter)."""
        a = np.asarray(audio, dtype=np.float32)[-self.window:]
        if len(a) < self.window:
            a = np.pad(a, (self.window - len(a), 0))
        logits = self.sess.run(None, {self.input_name: a[None]})[0][0]
        return float(softmax(logits.astype(np.float64))[self.wake_idx])

    def timed_score(self, audio: np.ndarray) -> tuple[float, float]:
        t0 = time.perf_counter()
        p = self.score(audio)
        return p, (time.perf_counter() - t0) * 1000.0
