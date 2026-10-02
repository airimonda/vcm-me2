import json

import numpy as np
import pytest
import torch

import vcm.train as T
from vcm.data import PackedSplit
from vcm.labels import OOS_IDX


def _cfg(packs, **kw):
    cfg = {"pack_dir": str(packs), "max_train_clips": None, "seed": 0, "real_weight": 5.0, "oos_weight": 3.0,
           "use_negatives": True, "negatives_weight": 1.0}
    cfg.update(kw)
    return cfg


def test_train_set_without_negatives_is_unchanged(fake_packs):
    ds, w, n_neg = T.build_train_set(_cfg(fake_packs, use_negatives=False))
    ref = PackedSplit(fake_packs, "train")
    assert isinstance(ds, PackedSplit) and n_neg == 0
    assert torch.equal(w, ref.sample_weights(5.0, 3.0))


def test_weights_align_with_concatenated_dataset(fake_packs):
    gold = PackedSplit(fake_packs, "train")
    neg = PackedSplit(fake_packs, "neg_train")
    ds, w, n_neg = T.build_train_set(_cfg(fake_packs, negatives_weight=2.5))
    assert n_neg == len(neg) == 20 and len(ds) == len(gold) + len(neg) == len(w)
    gw = gold.sample_weights(5.0, 3.0)
    assert torch.equal(w[: len(gold)], gw)                          # gold rows untouched
    assert torch.allclose(w[len(gold):], torch.full((len(neg),), 2.5, dtype=torch.float64))
    # row i of the concatenation is the matching clip
    for i in (0, len(gold) - 1, len(gold), len(gold) + len(neg) - 1):
        wav, lab = ds[i]
        src = gold if i < len(gold) else neg
        j = i if i < len(gold) else i - len(gold)
        assert torch.equal(wav, src[j][0]) and torch.equal(lab, src[j][1])
    assert (ds[len(gold)][1][0] == OOS_IDX).item()
    assert wav.shape[0] == 96000


def test_missing_neg_train_falls_back(fake_packs, tmp_path):
    for n in ("train_audio.npy", "train_meta.parquet"):
        (tmp_path / n).symlink_to(fake_packs / n)
    ds, w, n_neg = T.build_train_set(_cfg(tmp_path))
    assert n_neg == 0 and len(ds) == len(w)


def test_train_cli_overrides_and_logging(fake_packs, tmp_path):
    out = tmp_path / "run"
    base = ["--arch", "tc_resnet", "--tier", "S", "--epochs", "2", "--batch-size", "4", "--device", "cpu",
            "--aug", "none", "--pack-dir", str(fake_packs)]
    s = T.main(base + ["--out", str(out), "--misfire-weight", "1.0", "--use-negatives", "--negatives-weight", "2.0"])
    rows = [json.loads(l) for l in (out / "log.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for r in rows:
        assert 0.0 <= r["test_neg_misfire"] <= 1.0 and 0.0 <= r["test_neg_misfire_tau"] <= 1.0
        assert r["train_loss_misfire"] > 0.0
        assert abs(r["train_loss"] - (r["train_loss_cmd"] + r["train_loss_slot"] + r["train_loss_misfire"])) < 1e-5
    import yaml
    cfg = yaml.safe_load((out / "config.yaml").read_text())
    assert cfg["misfire_weight"] == 1.0 and cfg["use_negatives"] is True and cfg["negatives_weight"] == 2.0
    assert "neg_misfire" in s["best"]
    # defaults: features off, old numbers unchanged, but neg_test still scored
    out2 = tmp_path / "plain"
    T.main(base + ["--out", str(out2)])
    r2 = [json.loads(l) for l in (out2 / "log.jsonl").read_text().splitlines()]
    assert all(r["train_loss_misfire"] == 0.0 for r in r2) and "test_neg_misfire" in r2[0]
    assert yaml.safe_load((out2 / "config.yaml").read_text())["use_negatives"] is False


def test_neg_scoring_does_not_change_gold_metrics(fake_packs, tmp_path):
    """Same run with and without the neg_test pack present: identical gold curve."""
    plain = tmp_path / "packs_no_neg"
    plain.mkdir()
    for n in ("train_audio.npy", "train_meta.parquet", "test_audio.npy", "test_meta.parquet"):
        (plain / n).symlink_to(fake_packs / n)
    args = ["--arch", "tc_resnet", "--tier", "S", "--epochs", "2", "--batch-size", "4", "--device", "cpu",
            "--aug", "none"]
    T.main(args + ["--pack-dir", str(fake_packs), "--out", str(tmp_path / "a")])
    T.main(args + ["--pack-dir", str(plain), "--out", str(tmp_path / "b")])
    la = [json.loads(l) for l in (tmp_path / "a" / "log.jsonl").read_text().splitlines()]
    lb = [json.loads(l) for l in (tmp_path / "b" / "log.jsonl").read_text().splitlines()]
    assert "test_neg_misfire" in la[0] and "test_neg_misfire" not in lb[0]
    for x, y in zip(la, lb):
        for k in ("train_loss", "test_select_score", "test_variation_bal_acc", "test_oos_false_accept"):
            assert x[k] == y[k]


def test_eval_neg_test_and_tau_table(fake_packs, tmp_path):
    from vcm.eval import evaluate_model
    out = tmp_path / "run"
    T.main(["--arch", "tc_resnet", "--tier", "S", "--epochs", "1", "--batch-size", "4", "--device", "cpu",
            "--aug", "none", "--pack-dir", str(fake_packs), "--out", str(out)])
    m = evaluate_model(out / "best.pt", fake_packs, "neg_test", progress=False, out_dir=tmp_path / "ev")
    assert m["n"] == 10 and 0.0 <= m["misfire_rate"] <= 1.0
    assert set(m["by_neg_kind"]) == {"noise_only", "babble", "reversed", "truncated", "near_silence"}
    assert abs(m["misfire_rate"] - m["oos_false_accept"]) < 1e-9          # every neg clip is OOS
    assert "neg_kind" in open(tmp_path / "ev" / "predictions.csv").readline()
    m = evaluate_model(out / "best.pt", fake_packs, "test", progress=False, tau_table=True, out_dir=tmp_path / "ev2")
    tab = m["tau_table"]
    assert {"tau", "variation_bal_acc", "oos_false_accept", "neg_misfire", "in_scope_false_reject"} <= set(tab[0])
    assert tab[0]["tau"] == 0.0 and tab[-1]["tau"] > 0.9
    assert (tmp_path / "ev2" / "tau_sweep.csv").exists()
    # tau_table is monotone: raising tau can only reject more
    mf = [r["neg_misfire"] for r in tab]
    assert all(a >= b - 1e-12 for a, b in zip(mf, mf[1:]))


def test_collect_results_has_neg_column(tmp_path):
    from collect_results import collect
    d = tmp_path / "r_tag"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps({"arch": "a", "tier": "S", "seed": 0, "params": 1, "best_epoch": 1,
                                                "stop_epoch": 2, "stop_reason": "x",
                                                "best": {"neg_misfire": 0.25, "oos_false_accept": 0.1}}))
    df = collect(tmp_path)
    assert df["test_neg_misfire"].iloc[0] == 0.25 and df["run"].iloc[0] == "r_tag"
