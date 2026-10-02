"""int8 build of a wake model trained by train_wake.py (static QDQ, per-channel, log-mel front end kept fp32).

Calibration: --n-calib 1.5 s windows from data/wake/train.npz (half "Watson" placed as in training, half
negatives), seeded. Never tune clips.

  python scripts/wake_int8.py --fp32 exp/wake/ds_cnn_64/wake.onnx --out exp/wake/ds_cnn_64/wake_int8.onnx
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import train_wake as W  # noqa: E402


class Reader(CalibrationDataReader):
    def __init__(self, wins):
        self.it = iter([{"wav": w[None]} for w in wins])

    def get_next(self):
        return next(self.it, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default=str(ROOT / "data/wake/train.npz"))
    ap.add_argument("--n-calib", type=int, default=400)
    a = ap.parse_args()
    tr = W.Clips(a.data)
    rng = np.random.default_rng(0)
    pos, neg = np.nonzero(tr.label == 1)[0], np.nonzero(tr.label != 1)[0]
    wins = []
    for i in rng.choice(pos, a.n_calib // 2, replace=False):
        b = W.place(tr.clip(i), tr.onset[i], tr.end[i], tr.lead[i], rng, True)
        wins.append(b[W.PAD: W.PAD + W.WIN])
    for i in rng.choice(neg, a.n_calib - a.n_calib // 2, replace=False):
        b = W.place(tr.clip(i), 0, 0, 0, rng, False)
        wins.append(b[W.PAD: W.PAD + W.WIN])
    wins = [w.astype(np.float32) / 32768.0 for w in wins]
    excl = [n.name for n in onnx.load(a.fp32).graph.node if n.name.startswith("/frontend/")]
    quantize_static(a.fp32, a.out, Reader(wins), quant_format=QuantFormat.QDQ, per_channel=True,
                    activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                    op_types_to_quantize=["Conv", "MatMul", "Gemm"], nodes_to_exclude=excl)
    print(a.out, Path(a.fp32).stat().st_size, "->", Path(a.out).stat().st_size, "bytes")


if __name__ == "__main__":
    main()
