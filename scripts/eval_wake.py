"""Score wake-word ONNX models on data/wake/tune.npz exactly as train_wake.py selects them.

Same clip-level recall (clean and seeded playback-chain + noise versions, any 0.25 s-hop window >= threshold)
and false wakes per hour on the tune negative stream, for every threshold. For each model: the row at its
own threshold (--at, or the sidecar's) and the lowest threshold within --fa-budget.

  python scripts/eval_wake.py --models models/wake/vcm_wake_int8.onnx exp/wake/ds_cnn_64/wake.onnx --at 0.55 auto
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))
import train_wake as W  # noqa: E402
from vcm.augment import NoiseBank  # noqa: E402


class OnnxModel:
    def __init__(self, path, threads=4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.dynamic = not isinstance(self.sess.get_inputs()[0].shape[0], int)

    def __call__(self, x):
        x = x.cpu().numpy()
        if self.dynamic:
            return torch.from_numpy(self.sess.run(None, {"wav": x})[0])
        return torch.from_numpy(np.concatenate([self.sess.run(None, {"wav": r[None]})[0] for r in x]))

    def eval(self):
        return self

    def train(self, *_):
        return self


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--at", nargs="+", default=None, help="threshold per model, or 'auto' (lowest within budget)")
    ap.add_argument("--data", default=str(ROOT / "data/wake/tune.npz"))
    ap.add_argument("--noise", default=str(ROOT / "data/packs/noise.npz"))
    ap.add_argument("--fa-budget", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nb = NoiseBank(a.noise, device=dev) if Path(a.noise).exists() else None
    tune = W.TuneSet(W.Clips(a.data), dev, nb)
    tune.device = "cpu"
    th = W.THRESHOLDS
    res = {}
    for i, p in enumerate(a.models):
        at = (a.at[i] if a.at and i < len(a.at) else "auto")
        if at != "auto":
            th_i = np.unique(np.append(th, float(at)))
        else:
            th_i = th
        _, pick, rows, maxp = tune.score(OnnxModel(p), th_i, a.fa_budget)
        row = pick if at == "auto" else next(r for r in rows if abs(r["threshold"] - float(at)) < 1e-9)
        per_src = {s: {k: float((m[tune.pos_src == s] >= row["threshold"]).mean()) for k, m in maxp.items()}
                   for s in sorted(set(tune.pos_src))}
        res[p] = {"at": row, "budget_pick": pick, "recall_by_source": per_src}
        print(p, json.dumps(res[p]), flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
