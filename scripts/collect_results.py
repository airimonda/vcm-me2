"""Collect exp/*/summary.json into results/bakeoff.csv."""
import argparse
import json
from pathlib import Path

import pandas as pd

COLS = ["arch", "tier", "seed", "params", "best_epoch", "stop_epoch", "stop_reason",
        "test_select_score", "test_variation_bal_acc", "test_real_variation_bal_acc", "command_acc", "oos_false_accept",
        "test_neg_misfire", "run"]


def collect(exp: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(exp.glob("*/summary.json")):
        s = json.loads(p.read_text())
        b = s.get("best") or {}
        rows.append({"arch": s["arch"], "tier": s["tier"], "seed": s["seed"], "params": s["params"],
                     "best_epoch": s["best_epoch"], "stop_epoch": s["stop_epoch"], "stop_reason": s["stop_reason"],
                     "test_select_score": b.get("select_score"), "test_variation_bal_acc": b.get("variation_bal_acc"),
                     "test_real_variation_bal_acc": b.get("real_variation_bal_acc"), "command_acc": b.get("command_acc"),
                     "oos_false_accept": b.get("oos_false_accept"),
                     # synthetic-negative misfire rate at the best epoch (None if no neg_test pack was present)
                     "test_neg_misfire": b.get("neg_misfire"), "run": p.parent.name})
    return pd.DataFrame(rows, columns=COLS)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp")
    ap.add_argument("--out", default="results/bakeoff.csv")
    a = ap.parse_args()
    df = collect(Path(a.exp))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(df.to_string(index=False))
