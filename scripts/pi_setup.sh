#!/usr/bin/env bash
# One-time setup of the Raspberry Pi (Raspberry Pi OS Bookworm/Trixie, 64-bit) for the VCM-ME2 runtime.
# Run it ON the Pi, from the synced repo:   cd ~/vcm-me2 && bash scripts/pi_setup.sh [--service] [--kiosk] [--raspotify] [--echo-cancel]
#
#   (no flag)     apt packages + .venv with the Python dependencies
#   --echo-cancel install the PipeWire WebRTC echo-cancel config (then set the defaults with wpctl, see the file)
#   --raspotify   install raspotify (Spotify Connect) and name the device (LIBRESPOT_NAME, default "Watson")
#   --service     install + enable the systemd user service (deploy/vcm-me2.service) and linger
#   --kiosk       autostart Chromium in kiosk mode on the dashboard (deploy/kiosk.desktop)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
DEVICE_NAME="${LIBRESPOT_NAME:-Watson}"

DO_SERVICE=0; DO_KIOSK=0; DO_RASPOTIFY=0; DO_AEC=0
for a in "$@"; do
  case "$a" in
    --service) DO_SERVICE=1 ;; --kiosk) DO_KIOSK=1 ;; --raspotify) DO_RASPOTIFY=1 ;; --echo-cancel) DO_AEC=1 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done

echo "== apt packages"
sudo apt-get update
# python3-venv/pip: the venv; libportaudio2: sounddevice; libsndfile1: soundfile; mpv/pipewire-*/wireplumber: reply and
# earcon playback + echo cancel; alsa-utils: arecord fallback; espeak-ng: TTS fallback when a reply clip is missing;
# pulseaudio-utils: pactl; chromium: kiosk dashboard; curl: raspotify installer; bluez: Bluetooth speaker
sudo apt-get install -y python3-venv python3-pip libportaudio2 libsndfile1 mpv alsa-utils espeak-ng \
  pipewire pipewire-pulse pipewire-audio wireplumber libspa-0.2-modules pulseaudio-utils \
  bluez curl unclutter ffmpeg
sudo apt-get install -y chromium-browser || sudo apt-get install -y chromium

echo "== python venv"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install numpy onnxruntime sounddevice aiohttp requests pyyaml soundfile psutil
# scipy is optional (only for resampling non-16 kHz WAVs); the runtime falls back to numpy interpolation without it.

if [ ! -f config/spotify.json ]; then
  echo "NOTE: config/spotify.json is missing. Create it with:  .venv/bin/python scripts/spotify_auth.py   (see docs/runtime.md)"
fi

if [ "$DO_AEC" = 1 ]; then
  echo "== PipeWire echo cancellation"
  mkdir -p ~/.config/pipewire/pipewire.conf.d
  cp deploy/pipewire-echo-cancel.conf ~/.config/pipewire/pipewire.conf.d/60-echo-cancel.conf
  systemctl --user restart pipewire pipewire-pulse wireplumber || true
  echo "Now make echo-cancel-source / echo-cancel-sink the defaults:  wpctl status ; wpctl set-default <id>"
fi

if [ "$DO_RASPOTIFY" = 1 ]; then
  echo "== raspotify (Spotify Connect) as '$DEVICE_NAME'"
  curl -sL https://dtcooper.github.io/raspotify/install.sh | sh
  sudo sed -i '/^#\?LIBRESPOT_NAME=/d' /etc/raspotify/conf
  echo "LIBRESPOT_NAME=\"$DEVICE_NAME\"" | sudo tee -a /etc/raspotify/conf >/dev/null
  sudo systemctl enable --now raspotify
  echo "raspotify runs as its own service; to play through the echo-cancel sink see docs/runtime.md"
fi

if [ "$DO_SERVICE" = 1 ]; then
  echo "== systemd user service"
  mkdir -p ~/.config/systemd/user
  cp deploy/vcm-me2.service ~/.config/systemd/user/
  systemctl --user daemon-reload
  systemctl --user enable --now vcm-me2
  sudo loginctl enable-linger "$USER"
fi

if [ "$DO_KIOSK" = 1 ]; then
  echo "== kiosk autostart"
  mkdir -p ~/.config/autostart
  cp deploy/kiosk.desktop ~/.config/autostart/kiosk.desktop
fi

echo "== smoke test (no mic, no sound): one clip through the model"
.venv/bin/python -c "
from runtime.model import CommandModel; import numpy as np
m = CommandModel('models/vcm_conformer_M_ens3.onnx'); print('command model ok, tau =', m.tau)
from runtime.model import WakeModel
w = WakeModel('models/wake/vcm_wake_int8.onnx'); print('wake model ok, window =', w.window)
" || echo "model smoke test failed: were models/ synced?"
echo "done. Try:  .venv/bin/python scripts/pi_runtime.py --mock all --no-mic    (dashboard on :8080)"
