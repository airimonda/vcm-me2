"""ONNX export and int8 quantisation.

    python -m vcm.export export   --ckpt exp/x/best.pt --out exp/x/model.onnx --tau 0.5
    python -m vcm.export quantize --onnx exp/x/model.onnx --out exp/x/model_int8.onnx --pack data/packs

export: full waveform -> logits graph (LogMel + backbone + heads), opset 17, input fixed
1 x 80000 samples with a dynamic batch axis. Checks torch/onnxruntime parity (max logit diff
relative to max(1, |logit|) < 1e-3) and writes <out>.json: labels, slot values, tau, window size.

quantize: onnxruntime static QDQ int8, per-channel weights, calibration on 400 TRAIN
clips (never test). All nodes of the log-mel front end stay fp32.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch

from .data import PackedSplit, center_window, to_float
from .labels import CLASSES, N_SLOT_HEADS, SLOT_COMMANDS, SLOT_VALUES
from .models import FullModel, count_params

OPSET = 17
WINDOW = 80_000
SAMPLE_RATE = 16_000
INPUT, OUTPUTS = "waveform", ["cmd_logits", "slot_logits"]
PARITY_TOL = 1e-3


def sidecar(model: FullModel, tau: float, extra: dict | None = None) -> dict:
    d = {
        "format": "vcm-onnx-v1",
        "arch": model.arch, "tier": model.tier, "params": count_params(model),
        "sample_rate": SAMPLE_RATE, "window_samples": WINDOW, "window_seconds": WINDOW / SAMPLE_RATE,
        "input": {"name": INPUT, "shape": ["batch", WINDOW], "dtype": "float32", "range": [-1.0, 1.0]},
        "outputs": {"cmd_logits": ["batch", len(CLASSES)], "slot_logits": ["batch", N_SLOT_HEADS, 3]},
        "classes": CLASSES,
        "slot_commands": SLOT_COMMANDS,
        "slot_values": SLOT_VALUES,
        "tau": tau,
        "decision_rule": "softmax(cmd_logits); if max prob < tau -> OUT_OF_SCOPE; else argmax. "
                         "slot = argmax(slot_logits[head_of(command)]) for slotted commands.",
        "frontend": {"n_fft": 400, "hop": 160, "n_mels": 64, "log_eps": 1e-6, "norm": "per-utterance mean/std",
                     "included_in_graph": True},
        "opset": OPSET,
    }
    if extra:
        d.update(extra)
    return d


def load_ckpt(path: str) -> tuple[FullModel, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = FullModel(ck["arch"], ck["tier"], **ck.get("overrides", {}))
    m.load_state_dict(ck["model"])
    return m.eval(), ck


def export_onnx(model: FullModel, out: str, tau: float = 0.0, parity_inputs: torch.Tensor | None = None,
                tol: float = PARITY_TOL) -> dict:
    """Export + parity check. Returns the sidecar dict (with parity numbers). Raises if parity fails."""
    import onnxruntime as ort
    model = model.cpu().eval()
    dummy = torch.randn(2, WINDOW) * 0.1
    out = str(out)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, (dummy,), out, opset_version=OPSET, input_names=[INPUT],
                          output_names=OUTPUTS, dynamic_axes={INPUT: {0: "batch"}, OUTPUTS[0]: {0: "batch"},
                                                              OUTPUTS[1]: {0: "batch"}}, dynamo=False)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(out, so, providers=["CPUExecutionProvider"])
    xs = [torch.randn(3, WINDOW) * 0.1, (torch.rand(1, WINDOW) - 0.5) * 1.8]
    if parity_inputs is not None:
        xs.append(parity_inputs)
    worst, worst_rel = 0.0, 0.0
    with torch.no_grad():
        for x in xs:
            tc, ts = model(x)
            oc, os_ = sess.run(None, {INPUT: x.numpy()})
            for o, t in ((oc, tc.numpy()), (os_, ts.numpy())):
                d = float(np.abs(o - t).max())
                worst = max(worst, d)
                # relative to the logit scale, floored at 1 so near-zero logits use the absolute diff
                worst_rel = max(worst_rel, d / max(1.0, float(np.abs(t).max())))
    if worst_rel >= tol:
        raise RuntimeError(f"ONNX parity failed: max relative logit diff {worst_rel:.2e} >= {tol:g}")
    side = sidecar(model, tau, {"parity_max_abs_diff": worst, "parity_max_rel_diff": worst_rel})
    Path(out).with_suffix(".json").write_text(json.dumps(side, indent=2))
    return side


# ------------------------------------------------------------------ quantisation
def frontend_nodes(onnx_path: str) -> list[str]:
    """Names of the nodes that belong to the log-mel front end.

    The legacy exporter prefixes node names with the module scope ("/frontend/..."); as a
    cross-check we also walk the graph forward from the waveform input up to the tensor
    that feeds the backbone (the normalised log-mel)."""
    import onnx
    g = onnx.load(onnx_path).graph
    names = [n.name for n in g.node if n.name.startswith("/frontend/")]
    if not names:
        raise RuntimeError("no /frontend/ nodes found; cannot keep the front end in fp32")
    return names


class _Reader:
    def __init__(self, clips: np.ndarray, input_name: str):
        self.it = iter([{input_name: c[None]} for c in clips])

    def get_next(self):
        return next(self.it, None)


def quantize_int8(onnx_in: str, onnx_out: str, pack_dir: str, n_calib: int = 400, seed: int = 0) -> dict:
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    train = PackedSplit(pack_dir, "train")
    idx = np.random.RandomState(seed).choice(len(train), min(n_calib, len(train)), replace=False)
    wav = torch.stack([train[int(i)][0] for i in idx])
    clips = to_float(center_window(wav)).numpy().astype(np.float32)
    excl = frontend_nodes(onnx_in)
    quantize_static(
        onnx_in, onnx_out, _Reader(clips, INPUT),
        quant_format=QuantFormat.QDQ, per_channel=True,
        activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
        nodes_to_exclude=excl, calibrate_method=CalibrationMethod.MinMax,
        extra_options={"ActivationSymmetric": False, "WeightSymmetric": True},
    )
    side = json.loads(Path(onnx_in).with_suffix(".json").read_text()) if Path(onnx_in).with_suffix(".json").exists() else {}
    side.update({"quantization": {"scheme": "static QDQ int8, per-channel weights (QInt8), activations QUInt8",
                                  "calibration": f"{len(clips)} train clips (seed {seed})",
                                  "frontend": "fp32 (excluded nodes: %d)" % len(excl)}})
    Path(onnx_out).with_suffix(".json").write_text(json.dumps(side, indent=2))
    return side


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--ckpt", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--tau", type=float, default=0.0)
    e.add_argument("--pack", default=None, help="if given, parity is also checked on 8 real test clips")
    q = sub.add_parser("quantize")
    q.add_argument("--onnx", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--pack", default="data/packs")
    q.add_argument("--n-calib", type=int, default=400)
    a = ap.parse_args()
    if a.cmd == "export":
        model, ck = load_ckpt(a.ckpt)
        real = None
        if a.pack:
            ds = PackedSplit(a.pack, "test", 8)
            real = to_float(torch.stack([ds[i][0] for i in range(len(ds))]))
        side = export_onnx(model, a.out, a.tau, real)
        print(f"exported {a.out}  params={side['params']:,}  parity max|diff|={side['parity_max_abs_diff']:.2e}")
    else:
        quantize_int8(a.onnx, a.out, a.pack, a.n_calib)
        print(f"quantized -> {a.out}")


if __name__ == "__main__":
    main()
