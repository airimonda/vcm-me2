"""Paired sign test between two per-clip prediction CSVs (from `python -m vcm.eval --out DIR`).

    python scripts/compare.py results/a/predictions.csv results/b/predictions.csv

Clips are paired by `file`. Only clips where exactly one model is right count (ties are
dropped); two-sided exact binomial test on that split with p = 0.5.
"""
import argparse
import json

import pandas as pd
from scipy.stats import binomtest


def sign_test(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    m = a[["file", "correct"]].merge(b[["file", "correct"]], on="file", suffixes=("_a", "_b"))
    if len(m) == 0:
        raise SystemExit("no common clips")
    a_only = int(((m.correct_a == 1) & (m.correct_b == 0)).sum())
    b_only = int(((m.correct_a == 0) & (m.correct_b == 1)).sum())
    n = a_only + b_only
    p = binomtest(a_only, n, 0.5).pvalue if n else 1.0
    return {"n_common": len(m), "acc_a": float(m.correct_a.mean()), "acc_b": float(m.correct_b.mean()),
            "a_only_right": a_only, "b_only_right": b_only, "both_same": len(m) - n, "p_value": float(p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    args = ap.parse_args()
    print(json.dumps(sign_test(pd.read_csv(args.a), pd.read_csv(args.b)), indent=2))


if __name__ == "__main__":
    main()
