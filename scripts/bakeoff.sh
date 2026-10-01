#!/usr/bin/env bash
# Queue (arch, tier, seed) runs sequentially into exp/<arch>_<tier>_s<seed>/.
#  - finished runs (exp/<run>/summary.json exists) are skipped
#  - unfinished runs (last.pt exists) are resumed
# Then writes results/bakeoff.csv.
#
#   scripts/bakeoff.sh                          # round 1: all 6 archs, tier S, seed 0
#   scripts/bakeoff.sh runs.txt                 # file with lines "arch tier seed" (# comments ok)
#   EXTRA="--num-workers 4 --device cuda" scripts/bakeoff.sh
# Run it inside tmux; if the machine restarts just run it again.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
EXTRA="${EXTRA:-}"
LIST="${1:-}"

if [[ -n "$LIST" ]]; then
  RUNS=()
  while IFS= read -r l; do RUNS+=("$l"); done < <(grep -v '^\s*#' "$LIST" | grep -v '^\s*$')
else
  RUNS=()
  for a in ds_cnn bc_resnet tc_resnet matchbox crnn conformer; do RUNS+=("$a S 0"); done
fi

for line in "${RUNS[@]}"; do
  read -r arch tier seed <<<"$line"
  out="exp/${arch}_${tier}_s${seed}"
  if [[ -f "$out/summary.json" ]]; then
    echo "[skip] $out (finished)"; continue
  fi
  flag=""
  if [[ -f "$out/last.pt" ]]; then flag="--resume"; echo "[resume] $out"; else echo "[start] $out"; fi
  mkdir -p "$out"
  PYTHONPATH=. "$PY" -m vcm.train --arch "$arch" --tier "$tier" --seed "$seed" --out "$out" $flag $EXTRA \
    2>&1 | tee -a "$out/train.log"
done

PYTHONPATH=. "$PY" scripts/collect_results.py --exp exp --out results/bakeoff.csv
