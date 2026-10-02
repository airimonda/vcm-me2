"""The tune split: speaker-disjoint carve-out of TRAIN used for every choice; gold test / holdout stay untouched."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import make_tune_split as MT
import vcm.data as D
import vcm.eval as E
import vcm.train as T

ROOT = Path(__file__).resolve().parent.parent


def _meta(spec):
    """spec: list of (source, is_synthetic, accent, speaker_id, n_clips)."""
    rows = []
    for src, syn, acc, spk, n in spec:
        for k in range(n):
            rows.append({"file": f"audio/{src}_{spk}_{k}.wav", "speaker_id": spk, "source": src, "is_synthetic": syn,
                         "accent_group": acc, "variation_idx": k % 4})
    return pd.DataFrame(rows)


SPEC = ([("real_voice", 0, "Filipino (group)", f"r{i}", 20) for i in range(3)]
        + [("xela_A", 0, "Filipino (group)", f"x{i}", 6) for i in range(4)]
        + [("xela_B", 0, "Filipino (group)", "xb", 6)]
        + [("group_synthetic", 1, "Synthetic", f"s{i}", 8) for i in range(20)]
        + [("SLURP", 0, "Native English", f"u{i}", 5) for i in range(12)]
        + [("SNIPS", 0, "Native English", "lonely", 40)]                      # single speaker -> stays in train
        + [("SpeechCommands_v2", 0, "Unknown", None, 1)] * 3)                 # NaN ids: one pseudo speaker per clip


def test_speaker_disjoint_partition_and_strata():
    meta = _meta(SPEC)
    r = MT.build_split(meta, None, frac=0.15, seed=0, min_per_variation=0)
    tune, train = np.array(r["tune_train_idx"]), np.array(r["train_idx"])
    assert sorted(np.concatenate([tune, train]).tolist()) == list(range(len(meta)))
    spk = MT.speaker_keys(meta)
    assert not set(spk[tune]) & set(spk[train])                              # no speaker on both sides
    assert set(r["tune_speakers"]) == set(spk[tune])
    strata = r["summary"]["strata"]
    assert set(strata) == {"real_voice", "xela", "synthetic:Synthetic", "SLURP", "SNIPS", "SpeechCommands_v2"}
    for st, d in strata.items():
        if d["speakers"] > 1:
            assert d["tune_speakers"] >= 1 and d["tune_speakers"] < d["speakers"], st
        else:
            assert d["tune_clips"] == 0, st
    assert strata["SNIPS"]["tune_clips"] == 0
    assert strata["synthetic:Synthetic"]["tune_clips"] == 8 * 3              # 3 of 20 voices ~ 15%
    # deterministic given the seed, different for another seed
    assert MT.build_split(meta, None, 0.15, 0, 0)["tune_speakers"] == r["tune_speakers"]
    assert MT.build_split(meta, None, 0.15, 1, 0)["tune_speakers"] != r["tune_speakers"]


def test_variation_coverage_repair():
    meta = _meta(SPEC)
    meta["variation_idx"] = 93
    meta.loc[meta["speaker_id"] == "u5", "variation_idx"] = 7                 # variation 7 lives in one speaker only
    r = MT.build_split(meta, None, frac=0.1, seed=0, min_per_variation=3)
    assert "u5" in r["tune_speakers"]
    assert 7 not in r["summary"]["variations_below_min"]


def test_negatives_assignment_rule():
    meta = _meta(SPEC)
    r0 = MT.build_split(meta, None, 0.15, 0, 0)
    tune_files = [f for f, s in zip(meta["file"], MT.speaker_keys(meta)) if s in set(r0["tune_speakers"])]
    train_files = [f for f, s in zip(meta["file"], MT.speaker_keys(meta)) if s not in set(r0["tune_speakers"])]
    t1, t2, r1 = tune_files[0], tune_files[-1], train_files[0]
    neg = pd.DataFrame({"neg_kind": ["a"] * 6, "source_files": [
        f"train/{t1}", f"train/{t1};train/{t2}", f"train/{t1};noise/n1.wav", f"train/{r1};noise/n2.wav",
        f"train/{t1};train/{r1}", "noise/n3.wav"]})
    r = MT.build_split(meta, neg, 0.15, 0, 0)
    assert r["tune_speakers"] == r0["tune_speakers"]
    assert {0, 1, 2} <= set(r["neg_tune_idx"])                               # all train sources are tune speakers
    assert 3 in r["neg_train_idx"]                                           # only train speakers
    assert r["dropped_neg_idx"] == [4]                                       # mixed: dropped from both
    assert not set(r["neg_tune_idx"]) & set(r["neg_train_idx"])
    assert set(r["neg_tune_idx"]) | set(r["neg_train_idx"]) | set(r["dropped_neg_idx"]) == set(range(6))
    assert (5 in r["neg_tune_idx"]) == (MT._hash_frac(0, 5) < 0.15)          # noise only: seeded hash
    assert r["summary"]["negatives"]["dropped_mixed"] == 1


@pytest.fixture(scope="module")
def tune_json(fake_packs, tmp_path_factory):
    out = tmp_path_factory.mktemp("tune") / "tune_split.json"
    r = MT.main(["--pack-dir", str(fake_packs), "--frac", "0.3", "--seed", "0", "--min-per-variation", "0",
                 "--out", str(out)])
    assert len(r["tune_train_idx"]) > 0 and len(r["neg_tune_idx"]) + len(r["neg_train_idx"]) + len(r["dropped_neg_idx"]) == 20
    return out


def _args(packs, out, *extra):
    return ["--arch", "tc_resnet", "--tier", "S", "--epochs", "2", "--batch-size", "4", "--device", "cpu",
            "--aug", "none", "--pack-dir", str(packs), "--out", str(out), *extra]


def _log(out):
    return [json.loads(l) for l in (Path(out) / "log.jsonl").read_text().splitlines()]


def test_tune_training_never_touches_test_pack(fake_packs, tune_json, tmp_path, monkeypatch):
    packs = tmp_path / "packs"                                               # no test / neg_test pack at all
    packs.mkdir()
    for n in ("train_audio.npy", "train_meta.parquet", "neg_train_audio.npy", "neg_train_meta.parquet"):
        (packs / n).symlink_to(fake_packs / n)
    seen = []
    orig = D.PackedSplit.__init__

    def spy(self, pack_dir, split, *a, **k):
        seen.append(split)
        return orig(self, pack_dir, split, *a, **k)

    monkeypatch.setattr(D.PackedSplit, "__init__", spy)
    out = tmp_path / "run"
    s = T.main(_args(packs, out, "--select-split", "tune", "--tune-split", str(tune_json), "--use-negatives",
                     "--misfire-weight", "1.0"))
    assert set(seen) <= {"train", "neg_train"} and "test" not in seen and "neg_test" not in seen
    rows = _log(out)
    assert len(rows) == 2 and s["select_split"] == "tune"
    for r in rows:
        assert not any(k.startswith("test_") for k in r)
        assert {"tune_select_score", "tune_variation_bal_acc", "tune_real_variation_bal_acc", "tune_command_acc",
                "tune_oos_false_accept", "tune_neg_misfire", "tune_neg_misfire_tau"} <= set(r)
    # training set excludes the tune rows, the tune set is the tune rows
    t = D.load_tune_split(tune_json)
    ds, w, n_neg = T.build_train_set({"pack_dir": str(packs), "max_train_clips": None, "seed": 0, "real_weight": 5.0,
                                      "oos_weight": 3.0, "use_negatives": True, "negatives_weight": 1.0}, t)
    gold = ds.datasets[0]
    assert gold.idx.tolist() == t["train_idx"].tolist() and not set(gold.idx.tolist()) & set(t["tune_train_idx"].tolist())
    assert ds.datasets[1].idx.tolist() == t["neg_train_idx"].tolist() and n_neg == len(t["neg_train_idx"])
    assert (out / "best.pt").exists()


def test_tune_selection_uses_tune_subset(fake_packs, tune_json, tmp_path):
    out = tmp_path / "run"
    T.main(_args(fake_packs, out, "--select-split", "tune", "--tune-split", str(tune_json)))
    from vcm.eval import evaluate_model
    t = D.load_tune_split(tune_json)
    m = evaluate_model(out / "best.pt", fake_packs, "tune", progress=False, tune_split=str(tune_json))
    assert m["n"] == len(t["tune_train_idx"])
    best_log = [r for r in _log(out) if r["is_best"]][-1]
    assert abs(best_log["tune_variation_bal_acc"] - m["variation_bal_acc"]) < 1e-9     # same centre window, same subset


def test_default_select_split_is_unchanged_old_behaviour(fake_packs, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    T.main(_args(fake_packs, a))
    T.main(_args(fake_packs, b, "--select-split", "test"))
    la, lb = _log(a), _log(b)
    for x, y in zip(la, lb):
        assert {k: v for k, v in x.items() if k != "time_s"} == {k: v for k, v in y.items() if k != "time_s"}
    assert all("test_select_score" in r and not any(k.startswith("tune_") for k in r) for r in la)
    cfg = T.load_cfg(type("A", (), {"config": None})())
    assert cfg["select_split"] == "test"
    assert json.loads((a / "summary.json").read_text())["select_split"] == "test"


def test_eval_guard(fake_packs, tune_json, tmp_path, monkeypatch):
    out = tmp_path / "run"
    T.main(_args(fake_packs, out, "--select-split", "tune", "--tune-split", str(tune_json)))
    for split in ("test", "holdout", "neg_test"):
        with pytest.raises(PermissionError, match="final report only"):
            E.evaluate_model(out / "best.pt", fake_packs, split, progress=False)
    with pytest.raises(PermissionError):                                     # tau table on neg_test via the back door
        E.evaluate_model(out / "best.pt", fake_packs, "tune", progress=False, tau_table=True, neg_split="neg_test",
                         tune_split=str(tune_json))
    monkeypatch.setattr(sys, "argv", ["eval", "--model", str(out / "best.pt"), "--pack", str(fake_packs), "--split", "test"])
    with pytest.raises(SystemExit) as e:
        E.main()
    assert "test/holdout are for the final report only" in str(e.value)
    assert E.evaluate_model(out / "best.pt", fake_packs, "test", progress=False, final_test=True)["n"] == 12


def test_eval_tune_and_tau_table(fake_packs, tune_json, tmp_path):
    out = tmp_path / "run"
    T.main(_args(fake_packs, out, "--select-split", "tune", "--tune-split", str(tune_json)))
    t = D.load_tune_split(tune_json)
    m = E.evaluate_model(out / "best.pt", fake_packs, "neg_tune", progress=False, tune_split=str(tune_json))
    assert m["n"] == len(t["neg_tune_idx"]) and 0.0 <= m["misfire_rate"] <= 1.0
    m = E.evaluate_model(out / "best.pt", fake_packs, "tune", progress=False, tau_table=True, sweep=True,
                         tune_split=str(tune_json), out_dir=tmp_path / "ev")
    tab = m["tau_table"]
    assert {"tau", "variation_bal_acc", "oos_false_accept", "neg_misfire", "in_scope_false_reject"} <= set(tab[0])
    assert (tmp_path / "ev" / "tau_sweep.csv").exists()
    # tune is a subset of the train pack: the same rows as the training-time selection set
    ds = E.open_split(fake_packs, "tune", None, str(tune_json))
    assert ds.idx.tolist() == t["tune_train_idx"].tolist() and ds.split == "train"


def test_collect_results_tune_columns(tmp_path):
    from collect_results import collect
    for name, sel in (("old", None), ("new", "tune")):
        d = tmp_path / name
        d.mkdir()
        s = {"arch": "a", "tier": "S", "seed": 0, "params": 1, "best_epoch": 1, "stop_epoch": 2, "stop_reason": "x",
             "best": {"select_score": 0.5, "variation_bal_acc": 0.4, "neg_misfire": 0.25, "oos_false_accept": 0.1}}
        if sel:
            s["select_split"] = sel
        (d / "summary.json").write_text(json.dumps(s))
    df = collect(tmp_path).set_index("run")
    assert df.loc["old", "select_split"] == "test" and df.loc["old", "test_select_score"] == 0.5
    assert df.loc["new", "select_split"] == "tune" and df.loc["new", "tune_select_score"] == 0.5
    assert df.loc["new", "tune_neg_misfire"] == 0.25 and pd.isna(df.loc["new", "test_select_score"])
    assert "tune_real_variation_bal_acc" in df.columns and "tune_oos_false_accept" in df.columns
