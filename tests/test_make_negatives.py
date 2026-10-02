import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from fakedata import make_fake_dataset  # noqa: E402,F401  (tests dir is on sys.path via rootdir/conftest-less import)
import make_negatives as MN  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
KINDS = MN.KINDS


@pytest.fixture(scope="module")
def fake(tmp_path_factory):
    base = tmp_path_factory.mktemp("fake")
    return make_fake_dataset(base / "dataset", n_per_split=10)


@pytest.fixture(scope="module")
def negs(fake, tmp_path_factory):
    root, noise = fake
    out = tmp_path_factory.mktemp("neg") / "synthetic_negatives"
    MN.main(["--dataset", str(root), "--noise", str(noise), "--out", str(out), "--n-train", "50", "--n-test", "23",
             "--seed", "3"])
    return out


def _m(out, split):
    return pd.read_csv(out / split / "manifest.csv")


def test_counts_and_columns(negs, fake):
    gold_cols = list(pd.read_csv(fake[0] / "train" / "manifest.csv").columns)
    for split, n in (("train", 50), ("test", 23)):
        m = _m(negs, split)
        assert len(m) == n
        assert list(m.columns) == gold_cols + ["neg_kind", "source_files"]
        counts = m["neg_kind"].value_counts().to_dict()
        assert set(counts) == set(KINDS)
        assert max(counts.values()) - min(counts.values()) <= 1       # roughly equal shares
        assert (m["command"] == "OUT_OF_SCOPE").all() and (m["out_of_scope"] == 1).all()
        assert (m["bucket"] == "OUT_OF_SCOPE").all() and (m["source"] == "synthetic_negative").all()
        assert (m["is_synthetic"] == 1).all() and (m["accent_group"] == "Synthetic").all()
        assert m["variation"].isna().all() and m["slot_value"].isna().all()
        assert all(s.startswith(f"neg_{k}_") for s, k in zip(m["speaker_id"], m["neg_kind"]))
    assert (negs / "README.md").exists() and (negs / "summary.json").exists()
    assert "NOT part of the published gold splits" in (negs / "README.md").read_text()


def test_audio_format_and_duration(negs):
    for split in ("train", "test"):
        m = _m(negs, split)
        for f, d in zip(m["file"], m["duration_s"]):
            info = sf.info(str(negs / split / f))
            assert info.samplerate == 16000 and info.channels == 1 and info.subtype == "PCM_16"
            assert 1.0 - 1e-6 <= info.frames / 16000 <= 5.0 + 1e-6
            assert abs(info.frames / 16000 - d) < 1e-3


def test_split_isolation(negs, fake):
    root, noise = fake
    ntr, nte = MN.noise_split(sorted(noise.glob("*.wav")))
    assert ntr and nte and not ({f.name for f in ntr} & {f.name for f in nte})
    for split, other, noise_ok, noise_bad in (("train", "test", ntr, nte), ("test", "train", nte, ntr)):
        m = _m(negs, split)
        ok_noise = {f"noise/{f.name}" for f in noise_ok}
        bad_noise = {f"noise/{f.name}" for f in noise_bad}
        gold_speakers = set(pd.read_csv(root / split / "manifest.csv")["speaker_id"].astype(str))
        used = 0
        for kind, srcs, spk in zip(m["neg_kind"], m["source_files"].fillna(""), m["speaker_id"]):
            for s in filter(None, srcs.split(";")):
                used += 1
                assert not s.startswith(other + "/"), (split, s)
                assert s not in bad_noise
                assert s.startswith(split + "/") or s in ok_noise
                if s.startswith(split + "/"):
                    assert (root / s).exists()
            if kind in ("babble", "reversed", "truncated"):
                assert spk.split("_", 2)[2] in gold_speakers
            if kind == "near_silence":
                assert srcs == ""
        assert used > 0
    # kind specifics
    m = _m(negs, "train")
    assert (m.loc[m.neg_kind == "babble", "source_files"].str.count(";") + 1).between(3, 6).all()
    assert (m.loc[m.neg_kind.isin(["reversed"]), "source_files"].str.count(";") == 0).all()


def test_truncated_never_full_command(negs, fake):
    root, _ = fake
    for split in ("train", "test"):
        g = pd.read_csv(root / split / "manifest.csv").set_index("file")
        m = _m(negs, split)
        for srcs in m.loc[m.neg_kind == "truncated", "source_files"]:
            clip = srcs.split(";")[0][len(split) + 1:]
            assert g.loc[clip, "command"] != "OUT_OF_SCOPE"
            assert len(str(g.loc[clip, "transcript"]).split()) >= 3 and g.loc[clip, "duration_s"] >= 1.5


def test_levels(negs):
    m = _m(negs, "train")
    for f, k in zip(m["file"], m["neg_kind"]):
        x, _ = sf.read(str(negs / "train" / f), dtype="float32")
        db = MN.rms_db(x)
        if k == "noise_only":
            assert -46 < db < -14
        elif k == "babble":
            assert -31 < db < -19
        elif k == "near_silence":
            assert db < -49


def test_reversed_is_reversed(negs, fake):
    root, _ = fake
    m = _m(negs, "test")
    r = m[m.neg_kind == "reversed"].iloc[0]
    out, _ = sf.read(str(negs / "test" / r.file), dtype="int16")
    src = r.source_files.split(";")[0][len("test/"):]
    orig, _ = sf.read(str(root / "test" / src), dtype="int16")
    rev = orig[::-1]
    # the output is a (possibly trimmed) contiguous piece of the reversed source
    k = len(out)
    first = out[:200]
    hits = [s for s in range(len(rev) - k + 1) if np.array_equal(rev[s: s + 200], first)] if len(rev) >= k else []
    assert hits


def test_deterministic(fake, tmp_path):
    root, noise = fake
    outs = []
    for i, seed in enumerate((7, 7, 8)):
        o = tmp_path / f"o{i}"
        MN.main(["--dataset", str(root), "--noise", str(noise), "--out", str(o), "--n-train", "20", "--n-test", "10",
                 "--seed", str(seed)])
        outs.append(o)
    a, b, c = outs
    for split in ("train", "test"):
        assert (a / split / "manifest.csv").read_bytes() == (b / split / "manifest.csv").read_bytes()
        for f in _m(a, split)["file"]:
            assert (a / split / f).read_bytes() == (b / split / f).read_bytes()
    assert (a / "train" / "manifest.csv").read_bytes() != (c / "train" / "manifest.csv").read_bytes() or \
        any((a / "train" / f).read_bytes() != (c / "train" / f).read_bytes() for f in _m(a, "train")["file"])


def test_refuses_to_overwrite(negs, fake):
    root, noise = fake
    with pytest.raises(SystemExit):
        MN.main(["--dataset", str(root), "--noise", str(noise), "--out", str(negs)])


def test_readable_by_pack_data_and_dataset(negs, tmp_path):
    packs = tmp_path / "packs"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "pack_data.py"), "--negatives", str(negs),
                        "--out", str(packs), "--workers", "1"], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    from vcm.data import PackedSplit
    from vcm.labels import OOS_IDX
    for name, n, length in (("neg_train", 50, 96000), ("neg_test", 23, 80000)):
        ds = PackedSplit(packs, name)
        assert len(ds) == n
        wav, lab = ds[0]
        assert wav.shape[0] == length and wav.dtype.is_floating_point is False
        assert (ds.meta["cmd_idx"] == OOS_IDX).all()
        assert (ds.labels[:, 1:3] == -1).all() and (ds.meta["variation_idx"] == 93).all()
        assert set(ds.meta["neg_kind"]) == set(KINDS)
        assert (ds.meta["speech_s"] > 0).all()
