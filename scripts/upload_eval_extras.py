"""Publish the two remaining inputs of the pipeline as splits of the `synthetic_negatives` config on the Hub,
so scripts/reproduce.sh needs nothing but the public dataset:

  noise     the 88 DEMAND / MS-SNSD noise recordings used for augmentation (data/noise/*.wav)
  tune_oos  the 400 evaluation-only out-of-scope clips that set the reject threshold (data/tune_oos_src/tune_oos)

Both get the same columns as the config's train / test splits (empty where they do not apply). The dataset card's
YAML gets the two splits added to the config.

  python scripts/upload_eval_extras.py --dry-run     # build the parquet files only
  python scripts/upload_eval_extras.py               # build and upload (needs `hf auth login`)
"""
import argparse
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

REPO = "airimonda/ai231-me2-voice-commands"
COLS = ["file", "transcript", "command", "variation", "slot_value", "out_of_scope", "bucket", "speaker_id", "source",
        "is_synthetic", "accent_group", "numerals", "duration_s", "transcript_source", "variation_match",
        "whisper_check", "whisper_transcript", "note", "neg_kind", "source_files"]
INT = {"out_of_scope", "is_synthetic"}


def table(rows):
    audio = pa.array([{"bytes": r.pop("_bytes"), "path": os.path.basename(r["file"])} for r in rows],
                     type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
    arrays = [audio]
    for c in COLS:
        vals = [r.get(c) for r in rows]
        if c in INT:
            arrays.append(pa.array([int(v) if v not in (None, "") else 0 for v in vals], pa.int64()))
        elif c == "duration_s":
            arrays.append(pa.array([float(v) for v in vals], pa.float64()))
        else:
            arrays.append(pa.array([None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
                                    for v in vals], pa.string()))
    return pa.Table.from_arrays(arrays, names=["audio"] + COLS)


def noise_rows(d: Path):
    rows = []
    for f in sorted(d.glob("*.wav")):
        info = sf.info(str(f))
        src = "DEMAND" if f.name.startswith("demand_") else "MS-SNSD"
        rows.append({"_bytes": f.read_bytes(), "file": f"audio/{f.name}", "transcript": "<noise>",
                     "command": "OUT_OF_SCOPE", "out_of_scope": 1, "bucket": "OUT_OF_SCOPE",
                     "speaker_id": f"noise_{f.stem}", "source": src, "is_synthetic": 0, "accent_group": "Noise",
                     "duration_s": round(info.frames / info.samplerate, 3), "neg_kind": "noise_source",
                     "note": "augmentation noise; DEMAND: CC BY-SA 3.0, MS-SNSD: MIT (see the original releases)"})
    return rows


def tune_oos_rows(d: Path):
    m = pd.read_csv(d / "manifest.csv")
    rows = []
    for r in m.to_dict("records"):
        row = {c: r.get(c) for c in COLS if c in r}
        row["_bytes"] = (d / r["file"]).read_bytes()
        row["neg_kind"] = "tune_oos_" + str(r.get("oos_kind", ""))
        row["note"] = "evaluation only (reject threshold); real speech from speakers in no other split"
        rows.append(row)
    return rows


def patch_card(text: str, n_noise: int, n_oos: int, b_noise: int, b_oos: int) -> str:
    old = "  - split: test\n    path: synthetic_negatives/test-*\n"
    assert old in text, "synthetic_negatives config block not found in the card"
    if "synthetic_negatives/noise-*" not in text:
        text = text.replace(old, old + "  - split: noise\n    path: synthetic_negatives/noise-*\n"
                                       "  - split: tune_oos\n    path: synthetic_negatives/tune_oos-*\n")
    info_old = "  - name: test\n    num_bytes: 19874295.0\n    num_examples: 250\n"
    if info_old in text and "num_examples: %d\n" % n_noise not in text:
        text = text.replace(info_old, info_old + f"  - name: noise\n    num_bytes: {b_noise}.0\n    num_examples: {n_noise}\n"
                                                 f"  - name: tune_oos\n    num_bytes: {b_oos}.0\n    num_examples: {n_oos}\n")
    if "### Evaluation extras" not in text:
        text += ('\n\n### Evaluation extras (config `synthetic_negatives`, splits `noise` and `tune_oos`)\n\n'
                 f'* `noise` ({n_noise} files): the DEMAND and MS-SNSD background-noise recordings mixed into training '
                 'clips as augmentation (licences: DEMAND CC BY-SA 3.0, MS-SNSD MIT; see the original releases). '
                 'Not training examples.\n'
                 f'* `tune_oos` ({n_oos} clips): real out-of-scope speech (Common Voice general speech, SLURP and '
                 'Timers and Such near-miss requests) from speakers in no other split. Evaluation only: used with '
                 'the tune split to choose the reject threshold. Never trained on.\n\n'
                 'Both are read by `scripts/reproduce.sh` in https://github.com/airimonda/vcm-me2.\n')
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--noise", default="data/noise")
    ap.add_argument("--tune-oos", default="data/tune_oos_src/tune_oos")
    ap.add_argument("--out", default="data/hf_upload/synthetic_negatives")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tn, to = table(noise_rows(Path(a.noise))), table(tune_oos_rows(Path(a.tune_oos)))
    pn, po = out / "noise-00000-of-00001.parquet", out / "tune_oos-00000-of-00001.parquet"
    pq.write_table(tn, pn)
    pq.write_table(to, po)
    print(f"{pn}: {tn.num_rows} rows, {pn.stat().st_size / 1e6:.1f} MB")
    print(f"{po}: {to.num_rows} rows, {po.stat().st_size / 1e6:.1f} MB")
    from huggingface_hub import HfApi, hf_hub_download
    card = Path(hf_hub_download(REPO, "README.md", repo_type="dataset")).read_text()
    new = patch_card(card, tn.num_rows, to.num_rows, tn.nbytes, to.nbytes)
    (out.parent / "README.md").write_text(new)
    if a.dry_run:
        print("dry run: nothing uploaded; card preview at", out.parent / "README.md")
        return
    api = HfApi()
    for p in (pn, po):
        api.upload_file(path_or_fileobj=str(p), path_in_repo=f"synthetic_negatives/{p.name}", repo_id=REPO,
                        repo_type="dataset", commit_message=f"Add synthetic_negatives/{p.name}")
    api.upload_file(path_or_fileobj=str(out.parent / "README.md"), path_in_repo="README.md", repo_id=REPO,
                    repo_type="dataset", commit_message="Card: noise and tune_oos splits of synthetic_negatives")
    print("uploaded")


if __name__ == "__main__":
    main()
