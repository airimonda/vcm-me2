import numpy as np
import pandas as pd

from vcm.eval import compute_metrics, decide, tau_sweep, wilson
from vcm.labels import OOS_IDX


def _meta():
    rows = [
        # cmd, head, slot, var, synth, accent
        (13, 0, 0, 39, 1, "Synthetic"), (13, 0, 1, 40, 0, "A"), (0, -1, -1, 0, 0, "A"),
        (OOS_IDX, -1, -1, 93, 1, "Synthetic"), (OOS_IDX, -1, -1, 93, 0, "A"),
    ]
    return pd.DataFrame(rows, columns=["cmd_idx", "slot_head_idx", "slot_value_idx", "variation_idx",
                                       "is_synthetic", "accent_group"])


def _probs(pred_cmd, pred_slot):
    n = len(pred_cmd)
    cp = np.full((n, 20), 0.01)
    sp = np.full((n, 6, 3), 0.1)
    for i, (c, s) in enumerate(zip(pred_cmd, pred_slot)):
        cp[i, c] = 0.9
        if s >= 0:
            sp[i, 0, s] = 0.8
    return cp / cp.sum(1, keepdims=True), sp


def test_perfect_and_slot_error():
    meta = _meta()
    cp, sp = _probs([13, 13, 0, OOS_IDX, OOS_IDX], [0, 1, -1, -1, -1])
    m = compute_metrics(meta, cp, sp)
    assert m["variation_bal_acc"] == 1.0 and m["oos_false_accept"] == 0.0
    cp, sp = _probs([13, 13, 0, OOS_IDX, 3], [0, 2, -1, -1, -1])      # wrong slot, OOS accepted as command
    m = compute_metrics(meta, cp, sp)
    assert m["command_acc"] == 4 / 5
    assert abs(m["accuracy"] - 3 / 5) < 1e-9                            # slot error counts as wrong
    assert m["oos_false_accept"] == 0.5
    assert m["by_is_synthetic"]["0"]["n"] == 3


def test_tau_rejects():
    meta = _meta()
    cp, sp = _probs([13, 13, 0, 3, 3], [0, 1, -1, -1, -1])
    cp[3:] = 0.05
    cp[3, 3] = 0.3
    cp[4, 3] = 0.3
    cp /= cp.sum(1, keepdims=True)
    lo = compute_metrics(meta, cp, sp, tau=0.0)
    hi = compute_metrics(meta, cp, sp, tau=0.8)
    assert lo["oos_false_accept"] == 1.0 and hi["oos_false_accept"] == 0.0
    pc, _ = decide(cp, sp, 0.8)
    assert pc[3] == OOS_IDX
    df, best = tau_sweep(meta, cp, sp)
    assert best > 0.2 and len(df) > 5


def test_wilson():
    lo, hi = wilson(50, 100)
    assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    assert wilson(0, 0)[0] != wilson(0, 0)[0]


def test_sign_test():
    import sys
    sys.path.insert(0, "scripts")
    from compare import sign_test
    a = pd.DataFrame({"file": list("abcdefghij"), "correct": [1] * 10})
    b = pd.DataFrame({"file": list("abcdefghij"), "correct": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]})
    r = sign_test(a, b)
    assert r["a_only_right"] == 8 and r["b_only_right"] == 0 and r["p_value"] < 0.01
