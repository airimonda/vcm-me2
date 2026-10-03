"""Side-by-side table of reproduced vs published test metrics (used by scripts/reproduce.sh).

  python scripts/check_repro.py --repro repro/final --published results/final --names vcm_conformer_M_ens3_test
"""
import argparse
import json
import math
from pathlib import Path

KEYS = [("accuracy", "accuracy (command + slot)"), ("variation_bal_acc", "variation balanced acc."),
        ("real_variation_bal_acc", "... human voices"), ("command_acc", "command accuracy"),
        ("slot_acc", "slot accuracy"), ("oos_false_accept", "out of scope accepted"),
        ("in_scope_false_reject", "in scope rejected")]


def fmt(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repro", required=True)
    ap.add_argument("--published", required=True)
    ap.add_argument("--names", nargs="+", required=True)
    ap.add_argument("--tol", type=float, default=0.01, help="flag differences above this (other GPU / seed noise)")
    a = ap.parse_args()
    print("| run | metric | published | reproduced | diff |\n| --- | --- | ---: | ---: | ---: |")
    flagged = 0
    for n in a.names:
        r, p = Path(a.repro) / n / "metrics.json", Path(a.published) / n / "metrics.json"
        if not r.exists():
            print(f"| {n} | (missing reproduced metrics) | | | |")
            continue
        R = json.loads(r.read_text())
        P = json.loads(p.read_text()) if p.exists() else {}
        for k, lab in KEYS:
            rv, pv = R.get(k), P.get(k)
            if rv is None or (isinstance(rv, float) and math.isnan(rv)):
                continue
            d = rv - pv if isinstance(pv, (int, float)) and not math.isnan(pv) else None
            mark = " !" if d is not None and abs(d) > a.tol else ""
            flagged += bool(mark)
            print(f"| {n} | {lab} | {fmt(pv)} | {fmt(rv)} | {fmt(d)}{mark} |")
    print(f"\n{flagged} value(s) differ by more than {a.tol}; retraining on another GPU can move the last digits.")


if __name__ == "__main__":
    main()
