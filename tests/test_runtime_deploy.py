"""Deploy artefacts are present and sane (they are never run against the Pi here)."""
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("script", ["scripts/sync_to_pi.sh", "scripts/pi_setup.sh"])
def test_shell_scripts_parse(script):
    assert (ROOT / script).stat().st_mode & 0o111
    assert subprocess.run(["bash", "-n", str(ROOT / script)]).returncode == 0


def test_sync_excludes_the_big_and_private_things_and_targets_the_pi():
    s = (ROOT / "scripts/sync_to_pi.sh").read_text()
    for pat in ("/data/", "/exp/", ".venv/", "/results/runs/", "/runtime_logs/"):
        assert pat in s
    assert "abnunez@100.75.251.43" in s and "vcm-me2" in s and "--delete" not in s.replace("no --delete", "")


def test_pi_setup_installs_the_runtime_dependencies():
    s = (ROOT / "scripts/pi_setup.sh").read_text()
    for pkg in ("numpy", "onnxruntime", "sounddevice", "aiohttp", "requests", "pyyaml", "soundfile"):
        assert pkg in s
    assert "libportaudio2" in s and "espeak-ng" in s and "mpv" in s


def test_service_is_a_user_unit_with_the_new_paths():
    s = (ROOT / "deploy/vcm-me2.service").read_text()
    assert "%h/vcm-me2" in s and "scripts/pi_runtime.py" in s and "WantedBy=default.target" in s
    assert "User=" not in s and "/home/pi" not in s
    k = (ROOT / "deploy/kiosk.desktop").read_text()
    assert "theme=dark" in k and "localhost:8080" in k and "--kiosk" in k


def test_secrets_are_ignored_and_example_is_not():
    g = (ROOT / ".gitignore").read_text()
    assert "config/spotify.json" in g and "runtime_logs/" in g and "replies/" in g
    assert (ROOT / "config/spotify.example.json").exists()
    assert subprocess.run(["git", "check-ignore", "config/spotify.json"], cwd=ROOT).returncode == 0
    assert subprocess.run(["git", "check-ignore", "config/spotify.example.json"], cwd=ROOT).returncode == 1
