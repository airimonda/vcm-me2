import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(scope="session")
def fake_packs(tmp_path_factory):
    """Tiny fake gold dataset + synthetic negatives, all packed into one pack dir (train, test, neg_train, neg_test)."""
    from fakedata import make_fake_dataset
    import make_negatives as MN
    base = tmp_path_factory.mktemp("world")
    root, noise = make_fake_dataset(base / "dataset", n_per_split=12)
    negs = base / "dataset" / "synthetic_negatives"
    MN.main(["--dataset", str(root), "--noise", str(noise), "--out", str(negs), "--n-train", "20", "--n-test", "10"])
    packs = base / "packs"
    for extra in (["--dataset", str(root), "--splits", "train", "test"], ["--negatives", str(negs)]):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "pack_data.py"), *extra, "--out", str(packs),
                            "--workers", "1"], capture_output=True, text=True, cwd=ROOT)
        assert r.returncode == 0, r.stderr
    return packs
