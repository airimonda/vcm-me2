#!/usr/bin/env bash
# Copy the runtime (code + models + config + reply clips) from this repo to the Pi. Does NOT run anything on it.
#
#   scripts/sync_to_pi.sh                    # rsync to abnunez@100.75.251.43:~/vcm-me2
#   scripts/sync_to_pi.sh --dry-run          # show what would be copied
#   scripts/sync_to_pi.sh --no-secrets       # leave config/spotify.json out (create it on the Pi with spotify_auth.py)
#   PI_HOST=abnunez@vcm-pi.local PI_DIR=vcm-me2 scripts/sync_to_pi.sh
#   USE_TAR=1 scripts/sync_to_pi.sh          # tar-over-ssh instead of rsync (no rsync on one side)
#
# Nothing on the Pi is ever deleted (no --delete): its .venv, runtime_logs/ and config/spotify.json survive a sync.
# Excluded: data/, exp/, .venv, results/runs, runtime_logs, checkpoints (models/ckpt), .git, caches.
# models/ ships the command ensemble (fp32 + int8) and the wake models; reply clips from replies/ go along.
set -euo pipefail

PI_HOST="${PI_HOST:-abnunez@100.75.251.43}"
PI_DIR="${PI_DIR:-vcm-me2}"                    # relative to the Pi user's home
DRY=""
SECRETS=1
for a in "$@"; do
  case "$a" in
    --dry-run|-n) DRY="--dry-run" ;;
    --no-secrets) SECRETS=0 ;;
    -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done

cd "$(dirname "$0")/.."

EXCLUDES=(
  --exclude='.git/' --exclude='.venv/' --exclude='__pycache__/' --exclude='*.pyc' --exclude='.pytest_cache/'
  --exclude='/data/' --exclude='/exp/' --exclude='/results/runs/' --exclude='/runtime_logs/'
  --exclude='/models/ckpt/' --exclude='*.pt' --exclude='*.egg-info/' --exclude='.DS_Store'
  --exclude='/replies/_samples/' --exclude='/runtime/sounds/'
)
[ "$SECRETS" = 0 ] && EXCLUDES+=(--exclude='/config/spotify.json')

if [ "${USE_TAR:-0}" = 1 ]; then
  echo "tar over ssh -> $PI_HOST:~/$PI_DIR"
  [ -n "$DRY" ] && { echo "(dry run: tar mode only lists the files)"; tar -cf - "${EXCLUDES[@]}" . | tar -tf - | head -50; exit 0; }
  tar -czf - "${EXCLUDES[@]}" . | ssh "$PI_HOST" "mkdir -p ~/$PI_DIR && tar -xzf - -C ~/$PI_DIR"
else
  echo "rsync -> $PI_HOST:~/$PI_DIR"
  rsync -az --stats $DRY "${EXCLUDES[@]}" ./ "$PI_HOST:$PI_DIR/"
fi
echo "done. On the Pi:  cd ~/$PI_DIR && bash scripts/pi_setup.sh   (first time)  |  systemctl --user restart vcm-me2"
