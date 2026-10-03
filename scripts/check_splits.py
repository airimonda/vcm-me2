"""Check that the splits never share clips or speakers: tune vs the fitted part of train, and tune / tune_oos / train
vs test and holdout. Reads the pack metadata (data/packs/*_meta.parquet) and tune_split.json.

  python scripts/check_splits.py --pack-dir data/packs
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", default="data/packs")
    a = ap.parse_args()
    d = Path(a.pack_dir)
    tr = pd.read_parquet(d / "train_meta.parquet")
    te = pd.read_parquet(d / "test_meta.parquet")
    ho = pd.read_parquet(d / "holdout_meta.parquet")
    t = json.load(open(d / "tune_split.json"))
    tune, fit = tr.iloc[t["tune_train_idx"]], tr.iloc[t["train_idx"]]
    oos = pd.read_parquet(d / "tune_oos_meta.parquet") if (d / "tune_oos_meta.parquet").exists() else None
    spk = lambda m: set(m.speaker_id.astype(str))
    checks = {
        "tune rows also in the fitted rows": len(set(t["tune_train_idx"]) & set(t["train_idx"])),
        "speakers in tune and fit": len(spk(tune) & spk(fit)),
        "speakers in tune and test": len(spk(tune) & spk(te)),
        "speakers in tune and holdout": len(spk(tune) & spk(ho)),
        "speakers in train and test": len(spk(tr) & spk(te)),
        "speakers in train and holdout": len(spk(tr) & spk(ho)),
        "speakers in test and holdout": len(spk(te) & spk(ho)),
    }
    if oos is not None:
        checks.update({"speakers in tune_oos and train": len(spk(oos) & spk(tr)),
                       "speakers in tune_oos and test": len(spk(oos) & spk(te)),
                       "speakers in tune_oos and holdout": len(spk(oos) & spk(ho))})
    print(f"train {len(tr)} = fit {len(fit)} + tune {len(tune)}; test {len(te)}; holdout {len(ho)}"
          + (f"; tune_oos {len(oos)}" if oos is not None else ""))
    for k, v in checks.items():
        print(f"{'OK ' if v == 0 else 'BAD'} {k}: {v}")
    sys.exit(0 if all(v == 0 for v in checks.values()) else 1)


if __name__ == "__main__":
    main()
