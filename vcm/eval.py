"""Evaluation of a .pt checkpoint or .onnx model on a packed split.

Decision rule: command = argmax of the command head; if max softmax prob < tau the
clip is rejected (-> OUT_OF_SCOPE). Slot = argmax of the predicted command's head.
A clip is right if the command is right AND (for slotted commands) the slot is right.

Primary metric: variation balanced accuracy = mean recall over the 93 Option B
variations + the OOS group (94 groups).

    python -m vcm.eval --model exp/x/best.pt --pack data/packs --split test --tau 0.5 --out results/x
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import PackedSplit, center_window, to_float
from .labels import (CLASSES, CMD_OF_SLOT_HEAD, JOINT_NAMES, N_CMD, N_EVAL_GROUPS, OOS_IDX, SLOT_COMMANDS,
                      SLOT_HEAD_OF_CMD, joint_class)


# ------------------------------------------------------------------ predictors
def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_torch_model(path: str, device="cpu"):
    from .models import build_model
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = build_model(ck["arch"], ck["tier"], **ck.get("overrides", {}))
    m.load_state_dict(ck["model"])
    return m.to(device).eval(), ck


def make_predictor(path: str, device="cpu"):
    """Returns f(wav float32 np/torch (B,80000)) -> (cmd_logits (B,20), slot_logits (B,6,3)) as numpy."""
    if str(path).endswith(".onnx"):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name

        def f(wav):
            wav = wav.numpy() if isinstance(wav, torch.Tensor) else wav
            c, s = sess.run(None, {name: np.ascontiguousarray(wav, dtype=np.float32)})
            return c, s
        return f
    model, _ = load_torch_model(path, device)

    @torch.no_grad()
    def g(wav):
        wav = torch.as_tensor(wav).to(device)
        c, s = model(wav)
        return c.float().cpu().numpy(), s.float().cpu().numpy()
    return g


def predict_split(predict, ds: PackedSplit, batch_size: int = 32, progress: bool = False):
    """Run `predict` over a PackedSplit (centre 5 s window). Returns cmd_prob (N,20), slot_prob (N,6,3)."""
    n = len(ds)
    cp, sp = [], []
    rng = range(0, n, batch_size)
    if progress:
        from tqdm import tqdm
        rng = tqdm(rng, desc="eval")
    for i in rng:
        j = min(i + batch_size, n)
        wav = torch.stack([ds[k][0] for k in range(i, j)])
        wav = to_float(center_window(wav) if wav.shape[1] != 80000 else wav)
        c, s = predict(wav)
        cp.append(_softmax(c))
        sp.append(_softmax(s))
    return np.concatenate(cp), np.concatenate(sp)


# ------------------------------------------------------------------ decisions & metrics
def decide(cmd_prob, slot_prob, tau: float = 0.0):
    pred_cmd = cmd_prob.argmax(1)
    pred_cmd = np.where(cmd_prob.max(1) < tau, OOS_IDX, pred_cmd)
    pred_slot = np.full(len(pred_cmd), -1, dtype=np.int64)
    for c, h in SLOT_HEAD_OF_CMD.items():
        m = pred_cmd == c
        pred_slot[m] = slot_prob[m, h].argmax(1)
    return pred_cmd, pred_slot


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def correctness(meta: pd.DataFrame, pred_cmd, pred_slot):
    tc = meta["cmd_idx"].to_numpy()
    ts = meta["slot_value_idx"].to_numpy()
    cmd_ok = pred_cmd == tc
    slot_ok = np.where(ts >= 0, pred_slot == ts, True)
    return cmd_ok, cmd_ok & slot_ok


def _bal_recall(groups, ok, n_groups):
    rec = []
    for g in range(n_groups):
        m = groups == g
        if m.any():
            rec.append(ok[m].mean())
    return float(np.mean(rec)) if rec else float("nan")


def compute_metrics(meta: pd.DataFrame, cmd_prob, slot_prob, tau: float = 0.0, detail: bool = True):
    pred_cmd, pred_slot = decide(cmd_prob, slot_prob, tau)
    cmd_ok, ok = correctness(meta, pred_cmd, pred_slot)
    tc = meta["cmd_idx"].to_numpy()
    ts = meta["slot_value_idx"].to_numpy()
    var = meta["variation_idx"].to_numpy()
    oos = tc == OOS_IDX
    has_var = var >= 0                                    # excludes "other slot value" rows
    real = meta["is_synthetic"].astype(int).to_numpy() == 0

    out = {
        "tau": tau, "n": int(len(meta)), "n_oos": int(oos.sum()),
        "variation_bal_acc": _bal_recall(var[has_var], ok[has_var], N_EVAL_GROUPS),
        # same metric on real (non-synthetic) voices only; groups with no real clip are skipped
        "real_variation_bal_acc": _bal_recall(var[has_var & real], ok[has_var & real], N_EVAL_GROUPS),
        "accuracy": float(ok[has_var].mean()),
        "command_acc": float(cmd_ok.mean()),
        "command_bal_acc": _bal_recall(tc, cmd_ok, N_CMD),
        "oos_false_accept": float((pred_cmd[oos] != OOS_IDX).mean()) if oos.any() else float("nan"),
        "in_scope_false_reject": float((pred_cmd[~oos] == OOS_IDX).mean()) if (~oos).any() else float("nan"),
    }
    # slot heads: head argmax vs truth on clips whose true command owns that head
    sl_k = sl_n = 0
    heads = {}
    for h, name in enumerate(SLOT_COMMANDS):
        m = (tc == CMD_OF_SLOT_HEAD[h]) & (ts >= 0)
        if m.any():
            k = int((slot_prob[m, h].argmax(1) == ts[m]).sum())
            heads[name] = k / int(m.sum())
            sl_k += k
            sl_n += int(m.sum())
    out["slot_acc"] = sl_k / sl_n if sl_n else float("nan")
    out["slot_head_acc"] = heads
    lo, hi = wilson(int(ok[has_var].sum()), int(has_var.sum()))
    out["accuracy_ci95"] = [lo, hi]
    if detail:
        for col in ("accent_group", "is_synthetic"):
            d = {}
            for key, idx in meta.groupby(col).groups.items():
                pos = meta.index.get_indexer(idx)
                m = np.zeros(len(meta), bool)
                m[pos] = True
                m &= has_var
                if not m.any():
                    continue
                k, n = int(ok[m].sum()), int(m.sum())
                lo, hi = wilson(k, n)
                d[str(key)] = {"n": n, "accuracy": k / n, "ci95": [lo, hi],
                               "command_acc": float(cmd_ok[m].mean())}
            out["by_" + col] = d
    return out


def tau_sweep(meta, cmd_prob, slot_prob, taus=None, max_oos_fa=None):
    taus = np.round(np.concatenate([np.arange(0, 0.9, 0.05), np.arange(0.9, 1.0, 0.01)]), 2) if taus is None else taus
    rows = []
    for t in taus:
        m = compute_metrics(meta, cmd_prob, slot_prob, float(t), detail=False)
        rows.append({k: m[k] for k in ("tau", "variation_bal_acc", "command_acc", "oos_false_accept",
                                       "in_scope_false_reject")})
    df = pd.DataFrame(rows)
    cand = df if max_oos_fa is None else df[df["oos_false_accept"] <= max_oos_fa]
    if len(cand) == 0:
        cand = df
    best = cand.sort_values(["variation_bal_acc", "oos_false_accept"], ascending=[False, True]).iloc[0]
    return df, float(best["tau"])


def confusions(meta, pred_cmd, pred_slot):
    tc = meta["cmd_idx"].to_numpy()
    ts = meta["slot_value_idx"].to_numpy()
    cm = np.zeros((N_CMD, N_CMD), dtype=int)
    np.add.at(cm, (tc, pred_cmd), 1)
    keep = meta["variation_idx"].to_numpy() >= 0
    tj = np.array([joint_class(c, s) for c, s in zip(tc, ts)])
    pj = np.array([joint_class(c, s) for c, s in zip(pred_cmd, pred_slot)])
    cj = np.zeros((32, 32), dtype=int)
    np.add.at(cj, (tj[keep], pj[keep]), 1)
    return (pd.DataFrame(cm, index=CLASSES, columns=CLASSES),
            pd.DataFrame(cj, index=JOINT_NAMES, columns=JOINT_NAMES))


def predictions_frame(meta, cmd_prob, slot_prob, tau):
    pred_cmd, pred_slot = decide(cmd_prob, slot_prob, tau)
    _, ok = correctness(meta, pred_cmd, pred_slot)
    df = meta[["file", "command", "slot_value", "variation", "variation_idx", "speaker_id", "accent_group",
               "is_synthetic"]].copy()
    df["true_cmd_idx"] = meta["cmd_idx"].to_numpy()
    df["true_slot_idx"] = meta["slot_value_idx"].to_numpy()
    df["pred_cmd_idx"] = pred_cmd
    df["pred_command"] = [CLASSES[i] for i in pred_cmd]
    df["pred_slot_idx"] = pred_slot
    df["max_prob"] = cmd_prob.max(1)
    df["correct"] = ok.astype(int)
    return df


def evaluate_model(model_path, pack_dir, split="test", tau=0.0, max_clips=None, out_dir=None, device="cpu",
                   sweep=False, batch_size=32, progress=True, max_oos_fa=None):
    ds = PackedSplit(pack_dir, split, max_clips=max_clips)
    predict = make_predictor(model_path, device)
    cp, sp = predict_split(predict, ds, batch_size, progress)
    if sweep:
        sw, best_tau = tau_sweep(ds.meta, cp, sp, max_oos_fa=max_oos_fa)
        tau = best_tau
    m = compute_metrics(ds.meta, cp, sp, tau)
    m["model"] = str(model_path)
    m["split"] = split
    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "metrics.json").write_text(json.dumps(m, indent=2))
        predictions_frame(ds.meta, cp, sp, tau).to_csv(out / "predictions.csv", index=False)
        pc, sc = decide(cp, sp, tau)
        c1, c2 = confusions(ds.meta, pc, sc)
        c1.to_csv(out / "confusion_command.csv")
        c2.to_csv(out / "confusion_joint.csv")
        if sweep:
            sw.to_csv(out / "tau_sweep.csv", index=False)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help=".pt or .onnx")
    ap.add_argument("--pack", default="data/packs")
    ap.add_argument("--split", default="test", choices=["train", "test", "holdout"])
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--sweep-tau", action="store_true", help="sweep tau on this split and use the best")
    ap.add_argument("--max-oos-fa", type=float, default=None,
                    help="with --sweep-tau: only consider taus whose OOS false-accept rate is <= this")
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    m = evaluate_model(a.model, a.pack, a.split, a.tau, a.max_clips, a.out, a.device, a.sweep_tau, max_oos_fa=a.max_oos_fa)
    print(json.dumps({k: v for k, v in m.items() if not k.startswith("by_")}, indent=2))


if __name__ == "__main__":
    main()
