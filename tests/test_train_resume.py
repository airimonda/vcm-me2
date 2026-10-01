"""Interrupt a tiny CPU training run after epoch 1, resume, and require the same curve as an
uninterrupted run (needs data/packs)."""
import json
from pathlib import Path

import pytest

PACKS = Path("data/packs")
pytestmark = pytest.mark.skipif(not (PACKS / "test_meta.parquet").exists(), reason="packs not built")


def _args(out):
    return ["--arch", "tc_resnet", "--tier", "S", "--epochs", "3", "--max-train-clips", "128",
            "--max-test-clips", "64", "--device", "cpu", "--aug", "heavy", "--out", str(out),
            "--pack-dir", str(PACKS)]


def _log(out):
    return [json.loads(l) for l in (Path(out) / "log.jsonl").read_text().splitlines()]


def test_resume_matches_uninterrupted(tmp_path, monkeypatch):
    import vcm.train as T
    T.main(_args(tmp_path / "a"))

    calls = {"n": 0}
    real = T.score_test

    def boom(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt
        return real(*a, **k)

    monkeypatch.setattr(T, "score_test", boom)
    with pytest.raises(KeyboardInterrupt):
        T.main(_args(tmp_path / "b"))
    assert len(_log(tmp_path / "b")) == 1
    monkeypatch.setattr(T, "score_test", real)
    T.main(["--out", str(tmp_path / "b"), "--resume", "--device", "cpu"])
    la, lb = _log(tmp_path / "a"), _log(tmp_path / "b")
    assert [r["epoch"] for r in lb] == [1, 2, 3]
    for x, y in zip(la, lb):
        assert abs(x["train_loss"] - y["train_loss"]) < 1e-5
    assert (tmp_path / "b" / "best.pt").exists() and (tmp_path / "b" / "summary.json").exists()
