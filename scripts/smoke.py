"""Smoke run for every architecture: 1 epoch on 500 train clips (heavy aug), eval on 300 test
clips, ONNX export + parity, ONNX eval; int8 quantise + eval for --int8-arch.

    python scripts/smoke.py --tier S --device auto --out /tmp/vcm_smoke
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vcm.eval import evaluate_model  # noqa: E402
from vcm.export import export_onnx, load_ckpt, quantize_int8  # noqa: E402
from vcm.models import ARCHS  # noqa: E402
from vcm.train import main as train_main  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="S")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default="/tmp/vcm_smoke")
    ap.add_argument("--pack", default="data/packs")
    ap.add_argument("--int8-arch", default="bc_resnet")
    ap.add_argument("--archs", nargs="*", default=list(ARCHS))
    a = ap.parse_args()
    res = {}
    for arch in a.archs:
        out = Path(a.out) / f"{arch}_{a.tier}"
        r = {}
        try:
            t0 = time.time()
            s = train_main(["--arch", arch, "--tier", a.tier, "--epochs", "1", "--max-train-clips", "500",
                            "--max-test-clips", "300", "--aug", "heavy", "--out", str(out), "--device", a.device,
                            "--pack-dir", a.pack])
            r.update(params=s["params"], epoch_time_s=s["mean_epoch_time_s"], device=s["device"])
            m_pt = evaluate_model(out / "best.pt", a.pack, "train", 0.0, 300, progress=False)
            r["torch_eval_var_bal_acc"] = m_pt["variation_bal_acc"]
            model, _ = load_ckpt(out / "best.pt")
            side = export_onnx(model, str(out / "model.onnx"), tau=0.0)
            r["onnx_parity"] = side["parity_max_abs_diff"]
            m_ox = evaluate_model(out / "model.onnx", a.pack, "train", 0.0, 300, out_dir=out / "onnx_eval", progress=False)
            r["onnx_eval_var_bal_acc"] = m_ox["variation_bal_acc"]
            if arch == a.int8_arch:
                quantize_int8(str(out / "model.onnx"), str(out / "model_int8.onnx"), a.pack)
                m_q = evaluate_model(out / "model_int8.onnx", a.pack, "train", 0.0, 300, progress=False)
                r["int8_eval_var_bal_acc"] = m_q["variation_bal_acc"]
                r["int8_mb"] = (out / "model_int8.onnx").stat().st_size / 1e6
                r["fp32_mb"] = (out / "model.onnx").stat().st_size / 1e6
            r["status"] = "ok"
        except Exception as e:  # keep going so every arch is reported
            traceback.print_exc()
            r["status"] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"
        res[arch] = r
    print(json.dumps(res, indent=2))
    (Path(a.out) / "smoke_results.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
