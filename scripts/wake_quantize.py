"""Quantize the "Watson" wake-word ONNX model to int8 and compare it with fp32.

The wake model (models/wake/vcm_wake.onnx, input "wav" (1, 24000) = 1.5 s at 16 kHz, output "logits"
over ["WATSON", "OTHER"]) was trained in the earlier project and is reused unchanged. This script
  1. makes models/wake/vcm_wake_int8.onnx: ONNX Runtime static QDQ, per-channel int8, log-mel front end
     kept fp32 (nodes "/frontend/*"), calibrated on --n-calib clips of the wake manifest's train split;
  2. scores both models clip by clip the way the runtime does: sliding 1.5 s windows every 0.25 s,
     a clip fires if any window's WATSON probability >= --threshold.
Clips listed in the manifest but missing on disk are skipped and counted.

  python scripts/wake_quantize.py --manifest ~/voice-dataset-analysis/manifests/wake_labeled.csv
"""
import argparse
import json
import os

import numpy as np
import onnxruntime as ort
import pandas as pd
import soundfile as sf
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_static)

WIN, HOP = 24_000, 4_000


def load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    assert sr == 16000, (path, sr)
    return x


def windows(x):
    if len(x) < WIN:
        pad = WIN - len(x)
        x = np.pad(x, (pad // 2, pad - pad // 2))
    starts = range(0, len(x) - WIN + 1, HOP)
    return np.stack([x[s:s + WIN] for s in starts]).astype(np.float32)


class Reader(CalibrationDataReader):
    def __init__(self, paths):
        self.it = iter([{"wav": w[None]} for p in paths for w in windows(load(p))[:2]])

    def get_next(self):
        return next(self.it, None)


def session(path):
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def p_watson(sess, x):
    out = []
    for w in windows(x):
        z = sess.run(None, {"wav": w[None]})[0][0]
        e = np.exp(z - z.max())
        out.append(e[0] / e.sum())
    return max(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="~/voice-dataset-analysis/manifests/wake_labeled.csv")
    ap.add_argument("--fp32", default="models/wake/vcm_wake.onnx")
    ap.add_argument("--int8", default="models/wake/vcm_wake_int8.onnx")
    ap.add_argument("--n-calib", type=int, default=300)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--max-eval", type=int, default=1500, help="cap on dev clips scored (seeded sample)")
    ap.add_argument("--out", default="results/wake_quant.json")
    a = ap.parse_args()
    m = pd.read_csv(os.path.expanduser(a.manifest))
    m["exists"] = m.audio_path.map(os.path.exists)
    missing = m[~m.exists].groupby("split").size().to_dict()
    m = m[m.exists]

    tr = m[m.split == "train"]
    # balanced if WATSON train clips are on disk; otherwise (the Piper wake positives are gone) OTHER only
    pos, neg = tr[tr.label == "WATSON"], tr[tr.label == "OTHER"]
    k = min(len(pos), a.n_calib // 2)
    calib = pd.concat([pos.sample(k, random_state=0), neg.sample(a.n_calib - k, random_state=0)])
    import onnx
    excl = [n.name for n in onnx.load(a.fp32).graph.node if n.name.startswith("/frontend/")]
    quantize_static(a.fp32, a.int8, Reader(calib.audio_path.tolist()), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv", "MatMul", "Gemm"], nodes_to_exclude=excl)

    s32, s8 = session(a.fp32), session(a.int8)
    report = {"threshold": a.threshold, "calibration_watson_clips": int(k), "missing_clips": missing, "calibration_clips": len(calib),
              "fp32_mb": round(os.path.getsize(a.fp32) / 1e6, 3), "int8_mb": round(os.path.getsize(a.int8) / 1e6, 3)}
    for split in ("dev", "own"):
        d = m[m.split == split]
        if not len(d):
            continue
        if len(d) > a.max_eval:
            d = d.sample(a.max_eval, random_state=0)
        p32 = np.array([p_watson(s32, load(p)) for p in d.audio_path])
        p8 = np.array([p_watson(s8, load(p)) for p in d.audio_path])
        w = (d.label == "WATSON").to_numpy()
        f32, f8 = p32 >= a.threshold, p8 >= a.threshold
        report[split] = {
            "n_watson": int(w.sum()), "n_other": int((~w).sum()),
            "fp32_recall": float(f32[w].mean()) if w.any() else None,
            "int8_recall": float(f8[w].mean()) if w.any() else None,
            "fp32_false_wake": float(f32[~w].mean()), "int8_false_wake": float(f8[~w].mean()),
            "decisions_changed": int((f32 != f8).sum()), "max_abs_prob_diff": float(np.abs(p32 - p8).max())}
        print(split, json.dumps(report[split]), flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k not in ("dev", "own")}))


if __name__ == "__main__":
    main()
