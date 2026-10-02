"""Carve the `tune` split out of TRAIN, by speaker.

All choices (checkpoint, early stopping, hyper-parameters, reject threshold tau, misfire weight, ablations)
are made on `tune`. The gold test and holdout splits are never used for a choice.

    python scripts/make_tune_split.py --pack-dir data/packs --frac 0.15 --seed 0 --out data/packs/tune_split.json

Speakers are split, never clips: a speaker is either in `tune` or in `train_idx`.
Strata (each gets ~frac of its clips in tune; a stratum with a single speaker stays entirely in train):
  real_voice          the group's own recordings
  xela                every source starting with "xela" (Xela's recordings)
  synthetic:<accent>  synthetic voices (is_synthetic == 1), one stratum per accent_group
  <source>            each open-source dataset
Within a stratum the speakers are visited in seeded random order and one is added to tune when that moves the
tune clip count closer to frac * stratum clips (at least one speaker per multi-speaker stratum). Afterwards,
every variation (variation_idx 0..92) and OUT_OF_SCOPE (93) with fewer than --min-per-variation tune clips gets
whole extra speakers moved into tune (never emptying a stratum's train side), if some speaker can supply them.

Speakers without an id (NaN) are each their own speaker.

Negatives (neg_train_meta.parquet, optional): a row goes to `neg_tune` iff it has >= 1 train source clip
("train/<file>" entries of source_files; "noise/..." entries are ignored) and ALL of them belong to tune speakers;
to `neg_train` iff none does; rows with train sources on both sides are dropped (`dropped_neg_idx`). Rows without
any train source (noise only) go to neg_tune with probability ~frac by a seeded hash of the row index.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def speaker_keys(meta: pd.DataFrame) -> np.ndarray:
    """One key per row: speaker_id as str; missing ids become a per-clip pseudo speaker."""
    sid = meta["speaker_id"]
    return np.array([f"{src}|nan|{f}" if pd.isna(s) or str(s) == "" else str(s)
                     for s, src, f in zip(sid, meta["source"], meta["file"])])


def stratum_of(row_source: str, is_synth: int, accent: str) -> str:
    if str(row_source).startswith("xela"):
        return "xela"
    if int(is_synth) == 1:
        return f"synthetic:{accent}"
    return str(row_source)


def _hash_frac(seed: int, i: int) -> float:
    h = hashlib.sha256(f"tune-neg:{seed}:{i}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def pick_speakers(order, clips, frac):
    """Greedy: visit speakers in `order`, add one if it moves the clip count closer to the target."""
    total = sum(clips[s] for s in order)
    target = frac * total
    chosen, n = [], 0
    for s in order:
        if abs(n + clips[s] - target) < abs(n - target):
            chosen.append(s)
            n += clips[s]
    if not chosen:                               # at least one speaker in tune: the one closest to the target
        s = min(order, key=lambda k: (abs(clips[k] - target), k))
        chosen = [s]
    return chosen


def build_split(meta: pd.DataFrame, neg_meta: pd.DataFrame | None = None, frac: float = 0.15, seed: int = 0,
                min_per_variation: int = 3) -> dict:
    rng = np.random.RandomState(seed)
    spk = speaker_keys(meta)
    meta = meta.reset_index(drop=True)
    df = pd.DataFrame({"spk": spk, "stratum": [stratum_of(s, y, a) for s, y, a in
                                                zip(meta["source"], meta["is_synthetic"], meta["accent_group"])]})
    df["var"] = meta["variation_idx"].to_numpy()
    df["real"] = (meta["is_synthetic"].astype(int) == 0).to_numpy()
    # a speaker lives in exactly one stratum (majority stratum if the data ever disagrees)
    spk_stratum = df.groupby("spk")["stratum"].agg(lambda s: s.value_counts().index[0]).to_dict()
    df["stratum"] = df["spk"].map(spk_stratum)
    clips = df.groupby("spk").size().to_dict()

    by_stratum: dict[str, list[str]] = {}
    for s, st in sorted(spk_stratum.items()):
        by_stratum.setdefault(st, []).append(s)

    tune_spk: set[str] = set()
    for st in sorted(by_stratum):
        sp = by_stratum[st]
        if len(sp) < 2:
            continue
        order = [sp[i] for i in rng.permutation(len(sp))]
        tune_spk.update(pick_speakers(order, clips, frac))

    # make sure every variation / OOS has a few tune clips
    var_counts = lambda: np.bincount(df.loc[df["spk"].isin(tune_spk) & (df["var"] >= 0), "var"], minlength=94)
    moved = []
    for _ in range(200):
        vc = var_counts()
        short = [v for v in range(94) if vc[v] < min_per_variation]
        progress = False
        for v in short:
            need = min_per_variation - vc[v]
            sub = df[(df["var"] == v) & ~df["spk"].isin(tune_spk)]
            cands = []
            for s, n in sub.groupby("spk").size().items():
                st = spk_stratum[s]
                left = [x for x in by_stratum[st] if x not in tune_spk and x != s]
                if not left:
                    continue                                          # would empty the stratum's train side
                cands.append((-min(n, need), clips[s], rng.rand(), s))
            if cands:
                s = sorted(cands)[0][3]
                tune_spk.add(s)
                moved.append((v, s))
                progress = True
                break
        if not progress:
            break

    is_tune = df["spk"].isin(tune_spk).to_numpy()
    tune_idx = np.nonzero(is_tune)[0].tolist()
    train_idx = np.nonzero(~is_tune)[0].tolist()

    out = {"tune_train_idx": tune_idx, "train_idx": train_idx, "neg_tune_idx": [], "neg_train_idx": [],
           "dropped_neg_idx": [], "tune_speakers": sorted(tune_spk)}

    # ---- summary
    strata = {}
    for st, sp in sorted(by_stratum.items()):
        m = (df["stratum"] == st).to_numpy()
        strata[st] = {"speakers": len(sp), "tune_speakers": int(sum(s in tune_spk for s in sp)),
                      "clips": int(m.sum()), "tune_clips": int((m & is_tune).sum()),
                      "tune_frac": float((m & is_tune).sum() / max(m.sum(), 1))}
    vc = var_counts()
    real = df["real"].to_numpy()
    summary = {
        "frac": frac, "seed": seed, "n_train_clips": len(meta),
        "n_tune_clips": len(tune_idx), "n_train_idx": len(train_idx),
        "n_speakers": len(clips), "n_tune_speakers": len(tune_spk),
        "tune_frac_clips": len(tune_idx) / len(meta),
        "real_share_tune": float(real[is_tune].mean()) if is_tune.any() else float("nan"),
        "real_share_train": float(real[~is_tune].mean()) if (~is_tune).any() else float("nan"),
        "tune_oos_clips": int(((df["var"] == 93) & is_tune).sum()),
        "train_oos_clips": int(((df["var"] == 93) & ~is_tune).sum()),
        "per_variation_min_tune": int(vc.min()), "per_variation_min_idx": int(vc.argmin()),
        "variations_with_zero_tune": [int(v) for v in range(94) if vc[v] == 0],
        "variations_below_min": {int(v): int(vc[v]) for v in range(94) if vc[v] < min_per_variation},
        "min_per_variation": min_per_variation,
        "speakers_moved_for_variation_coverage": [(int(v), s) for v, s in moved],
        "strata": strata,
    }

    if neg_meta is not None:
        file_spk = dict(zip(meta["file"], spk))
        nt, nr, nd, unknown = [], [], [], 0
        for i, sf in enumerate(neg_meta["source_files"].fillna("").astype(str)):
            tr = []
            for e in (x for x in sf.split(";") if x):
                if e.startswith("train/"):
                    f = e[len("train/"):]
                    if f in file_spk:
                        tr.append(file_spk[f])
                    else:
                        unknown += 1
            if not tr:
                (nt if _hash_frac(seed, i) < frac else nr).append(i)
            else:
                flags = [s in tune_spk for s in tr]
                (nt if all(flags) else nd if any(flags) else nr).append(i)
        out.update(neg_tune_idx=nt, neg_train_idx=nr, dropped_neg_idx=nd)
        summary["negatives"] = {"rows": len(neg_meta), "neg_tune": len(nt), "neg_train": len(nr), "dropped_mixed": len(nd),
                                "unknown_train_sources": unknown}
        if "neg_kind" in neg_meta.columns:
            k = neg_meta["neg_kind"].astype(str).to_numpy()
            summary["negatives"]["neg_tune_by_kind"] = {x: int((k[nt] == x).sum()) for x in sorted(set(k))} if nt else {}
    out["summary"] = summary
    return out


def format_summary(s: dict) -> str:
    L = [f"tune split: {s['n_tune_clips']} / {s['n_train_clips']} train clips ({100 * s['tune_frac_clips']:.1f}%), "
         f"{s['n_tune_speakers']} / {s['n_speakers']} speakers (frac {s['frac']}, seed {s['seed']})",
         f"real share of clips: tune {100 * s['real_share_tune']:.1f}%  train {100 * s['real_share_train']:.1f}%",
         f"OUT_OF_SCOPE clips: tune {s['tune_oos_clips']}  train {s['train_oos_clips']}",
         f"per-variation tune clips (0..92 + OOS 93): min {s['per_variation_min_tune']} (variation_idx {s['per_variation_min_idx']}); "
         f"with 0: {s['variations_with_zero_tune'] or 'none'}; below {s['min_per_variation']}: {s['variations_below_min'] or 'none'}",
         f"speakers moved to tune for variation coverage: {len(s['speakers_moved_for_variation_coverage'])}",
         f"{'stratum':32s} {'spk':>9s} {'clips':>13s}  tune%"]
    for st, d in s["strata"].items():
        L.append(f"{st:32s} {d['tune_speakers']:4d}/{d['speakers']:<4d} {d['tune_clips']:6d}/{d['clips']:<6d} {100 * d['tune_frac']:5.1f}")
    if "negatives" in s:
        n = s["negatives"]
        L.append(f"negatives: {n['rows']} rows -> neg_tune {n['neg_tune']}, neg_train {n['neg_train']}, "
                 f"dropped (mixed) {n['dropped_mixed']}, unknown train sources {n['unknown_train_sources']}")
    else:
        L.append("negatives: no neg_train_meta.parquet in the pack dir (neg_*_idx left empty)")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack-dir", default="data/packs")
    ap.add_argument("--frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-per-variation", type=int, default=3)
    ap.add_argument("--out", default=None, help="default: <pack-dir>/tune_split.json")
    a = ap.parse_args(argv)
    pack = Path(a.pack_dir)
    meta = pd.read_parquet(pack / "train_meta.parquet")
    nm = pack / "neg_train_meta.parquet"
    neg = pd.read_parquet(nm) if nm.exists() else None
    res = build_split(meta, neg, a.frac, a.seed, a.min_per_variation)
    out = Path(a.out) if a.out else pack / "tune_split.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res))
    print(format_summary(res["summary"]))
    print(f"wrote {out}")
    return res


if __name__ == "__main__":
    main()
