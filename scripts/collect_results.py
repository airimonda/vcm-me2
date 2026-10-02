"""Collect exp/*/summary.json into results/bakeoff.csv."""
import argparse
import json
from pathlib import Path

import pandas as pd

# select_split says which split the best-epoch numbers were measured on: "tune" (speaker-disjoint slice of train) or
# "test" (old runs, default). The matching test_* / tune_* columns are filled, the other group stays empty.
METRICS = ["select_score", "variation_bal_acc", "real_variation_bal_acc", "command_acc", "oos_false_accept", "neg_misfire"]
COLS = (["arch", "tier", "seed", "params", "best_epoch", "stop_epoch", "stop_reason", "select_split"]
        + [f"test_{c}" for c in ("select_score", "variation_bal_acc", "real_variation_bal_acc")]
        + ["command_acc", "oos_false_accept", "test_neg_misfire"]
        + [f"tune_{c}" for c in ("select_score", "variation_bal_acc", "real_variation_bal_acc", "command_acc",
                                 "oos_false_accept", "neg_misfire")]
        + ["run"])


def collect(exp: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(exp.glob("*/summary.json")):
        s = json.loads(p.read_text())
        b = s.get("best") or {}
        sel = s.get("select_split", "test")          # summaries written before the tune split have no key
        row = {"arch": s["arch"], "tier": s["tier"], "seed": s["seed"], "params": s["params"],
               "best_epoch": s["best_epoch"], "stop_epoch": s["stop_epoch"], "stop_reason": s["stop_reason"],
               "select_split": sel, "command_acc": b.get("command_acc"), "oos_false_accept": b.get("oos_false_accept"),
               "run": p.parent.name}
        for k in METRICS:
            row[f"{sel}_{k}"] = b.get(k)             # neg_misfire is None if no negatives pack was present
        rows.append(row)
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
