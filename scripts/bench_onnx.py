"""Latency benchmark of exported VCM ONNX models on the target device (e.g. Raspberry Pi 4).

One inference = one 5 s window (1 x 80,000 samples, 16 kHz), front end included, as the runtime
would call it. Reports p50 / p95 / mean latency, real-time factor (latency / 5 s), peak RSS and
the CPU temperature, per model and per intra-op thread count. Only numpy + onnxruntime are needed.

  python bench_onnx.py --models vcm_conformer_M.onnx vcm_conformer_M_ens3.onnx --threads 1 2 4 --n 200
"""
import argparse
import json
import os
import resource
import subprocess
import time

import numpy as np
import onnxruntime as ort

WINDOW = 80_000


def temp_c():
    try:
        out = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True).stdout
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        return None


def bench(path, threads, n, warmup, x):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")   # idle CPU stays low on the Pi
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    for _ in range(warmup):
        sess.run(None, {name: x})
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        sess.run(None, {name: x})
        t.append((time.perf_counter() - t0) * 1000)
    t = np.array(t)
    return {"model": os.path.basename(path), "size_mb": round(os.path.getsize(path) / 1e6, 2), "threads": threads,
            "n": n, "p50_ms": round(float(np.percentile(t, 50)), 1), "p95_ms": round(float(np.percentile(t, 95)), 1),
            "mean_ms": round(float(t.mean()), 1), "rtf": round(float(np.percentile(t, 50)) / 5000, 4),
            "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1), "temp_c": temp_c()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rng = np.random.RandomState(0)
    x = (rng.randn(1, WINDOW) * 0.05).astype(np.float32)   # input content does not change the compute
    rows = []
    for m in a.models:
        for th in a.threads:
            r = bench(m, th, a.n, a.warmup, x)
            print(json.dumps(r), flush=True)
            rows.append(r)
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"onnxruntime": ort.__version__, "results": rows}, f, indent=1)


if __name__ == "__main__":
    main()
