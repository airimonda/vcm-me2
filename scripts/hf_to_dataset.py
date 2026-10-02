"""Turn the Hugging Face parquet release of the ME2 dataset into the folder layout
pack_data.py reads: <out>/<split>/manifest.csv + <out>/<split>/audio/*.wav, plus variations.csv.

  python scripts/hf_to_dataset.py --hf ~/hf-me2 --out ~/me2-dataset
  python scripts/pack_data.py --dataset ~/me2-dataset --splits train test holdout ...

Download the release first (airimonda/ai231-me2-voice-commands), e.g. with
huggingface_hub.snapshot_download(..., allow_patterns=["data/train-*", "data/test-*",
"data/holdout-*", "variations.csv"]). Only the default config is used; supplemental_synth is not.
"""
import argparse
import glob
import os
import shutil

import pandas as pd
import pyarrow.parquet as pq


def export_split(hf_dir: str, split: str, out_dir: str) -> int:
    files = sorted(glob.glob(os.path.join(hf_dir, "data", f"{split}-*.parquet")))
    if not files:
        raise SystemExit(f"no parquet files for split {split!r} in {hf_dir}/data")
    split_dir = os.path.join(out_dir, split)
    os.makedirs(os.path.join(split_dir, "audio"), exist_ok=True)
    rows = []
    for f in files:
        t = pq.read_table(f).to_pandas()
        for _, r in t.iterrows():
            name = os.path.basename(r["file"]) if r["file"] else os.path.basename(r["audio"]["path"])
            rel = os.path.join("audio", name)
            with open(os.path.join(split_dir, rel), "wb") as fh:
                fh.write(r["audio"]["bytes"])        # the release embeds the original 16 kHz WAV bytes
            row = r.drop(labels=["audio"]).to_dict()
            row["file"] = rel
            rows.append(row)
    m = pd.DataFrame(rows)
    if m["file"].duplicated().any():
        raise SystemExit(f"{split}: duplicate audio file names")
    m.to_csv(os.path.join(split_dir, "manifest.csv"), index=False)
    return len(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", required=True, help="local snapshot of the HF dataset repo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "test", "holdout"])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    shutil.copy(os.path.join(a.hf, "variations.csv"), os.path.join(a.out, "variations.csv"))
    for s in a.splits:
        print(s, export_split(a.hf, s, a.out), "clips")


if __name__ == "__main__":
    main()
