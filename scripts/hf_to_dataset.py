"""Download the Hugging Face release of the ME2 dataset and turn it into the folder layouts the pipeline reads.

  python scripts/hf_to_dataset.py --download --hf data/hf --out data/me2-dataset            # everything
  python scripts/hf_to_dataset.py --hf ~/hf-me2 --out ~/me2-dataset --splits train test holdout   # gold only

Outputs (each split folder = manifest.csv + audio/*.wav, what pack_data.py reads):
  <out>/{train,test,holdout}/         default config (gold splits) + <out>/variations.csv
  <out>/negatives/{train,test}/        config synthetic_negatives      -> pack_data.py --negatives <out>/negatives
  <out>/tune_oos/                      synthetic_negatives, split tune_oos -> pack_data.py --splits tune_oos
  <noise-out>/*.wav                    synthetic_negatives, split noise    -> pack_data.py --noise <noise-out>
supplemental_*, numerals are not used. The release embeds the original 16 kHz WAV bytes, so nothing is re-encoded.
"""
import argparse
import glob
import os
import shutil

import pandas as pd
import pyarrow.parquet as pq

REPO = "airimonda/ai231-me2-voice-commands"


def export(files, split_dir: str) -> int:
    """parquet rows (audio bytes + metadata) -> <split_dir>/audio/*.wav + manifest.csv"""
    os.makedirs(os.path.join(split_dir, "audio"), exist_ok=True)
    rows = []
    for f in files:
        t = pq.read_table(f).to_pandas()
        for _, r in t.iterrows():
            name = os.path.basename(r["file"]) if r.get("file") else os.path.basename(r["audio"]["path"])
            rel = os.path.join("audio", name)
            with open(os.path.join(split_dir, rel), "wb") as fh:
                fh.write(r["audio"]["bytes"])
            row = r.drop(labels=["audio"]).to_dict()
            row["file"] = rel
            rows.append(row)
    m = pd.DataFrame(rows)
    if m["file"].duplicated().any():
        raise SystemExit(f"{split_dir}: duplicate audio file names")
    m.to_csv(os.path.join(split_dir, "manifest.csv"), index=False)
    return len(m)


def parquet(hf_dir, sub, split):
    return sorted(glob.glob(os.path.join(hf_dir, sub, f"{split}-*.parquet")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="local snapshot of the HF dataset repo (written with --download)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test", "holdout"], help="gold splits to export")
    ap.add_argument("--no-negatives", action="store_true")
    ap.add_argument("--noise-out", default="data/noise")
    ap.add_argument("--download", action="store_true", help="fetch the needed files from the Hub first")
    a = ap.parse_args()
    if a.download:
        from huggingface_hub import snapshot_download
        pats = ["variations.csv", "synthetic_negatives/*"] + [f"data/{s}-*" for s in a.splits]
        snapshot_download(REPO, repo_type="dataset", local_dir=a.hf, allow_patterns=pats)
    os.makedirs(a.out, exist_ok=True)
    shutil.copy(os.path.join(a.hf, "variations.csv"), os.path.join(a.out, "variations.csv"))
    for s in a.splits:
        files = parquet(a.hf, "data", s)
        if not files:
            raise SystemExit(f"no parquet files for split {s!r} in {a.hf}/data")
        print(s, export(files, os.path.join(a.out, s)), "clips")
    if not a.no_negatives:
        for s in ("train", "test"):
            files = parquet(a.hf, "synthetic_negatives", s)
            if files:
                print("negatives", s, export(files, os.path.join(a.out, "negatives", s)), "clips")
    files = parquet(a.hf, "synthetic_negatives", "tune_oos")
    if files:
        print("tune_oos", export(files, os.path.join(a.out, "tune_oos")), "clips")
    else:
        print("tune_oos: not in this snapshot (the reject threshold stays at the published 0.40)")
    files = parquet(a.hf, "synthetic_negatives", "noise")
    if files:
        os.makedirs(a.noise_out, exist_ok=True)
        n = 0
        for f in files:
            for r in pq.read_table(f).to_pylist():
                with open(os.path.join(a.noise_out, os.path.basename(r["file"])), "wb") as fh:
                    fh.write(r["audio"]["bytes"])
                n += 1
        print("noise", n, "files ->", a.noise_out)
    else:
        print("noise: not in this snapshot (augmentation falls back to synthetic white / pink / brown noise)")


if __name__ == "__main__":
    main()
