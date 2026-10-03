#!/usr/bin/env bash
# One-command reproduction of the released voice command model from the public Hugging Face data.
#
#   bash scripts/reproduce.sh --check    # ~5 min, CPU: download test data, score the released ONNX, compare
#   bash scripts/reproduce.sh            # ~1 GPU-hour: rebuild packs, train the 3 seeds, export, score, compare
#   bash scripts/reproduce.sh --full     # ~5 GPU-hours: also both bake-off rounds, the ablations, the DS-CNN baseline
#
# Everything it writes goes to data/ (downloads, packs) and repro/ (runs, models, scores); the committed
# models/ and results/ are never overwritten. Compare table: repro/compare.md (vs results/final/*).
# Env: PY (python, default .venv/bin/python, created with uv if missing), NUM_WORKERS (default 4),
#      DEVICE (auto | cuda | mps | cpu), DATA_DIR (downloads, default data), PACK_DIR (default data/packs),
#      SMOKE=1 (pipeline test: 600 train clips, 2 epochs; numbers will not match).
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-train}"
case "$MODE" in --check) MODE=check ;; --full) MODE=full ;; train|"") MODE=train ;;
  *) echo "usage: bash scripts/reproduce.sh [--check | --full]" >&2; exit 2 ;; esac
PY="${PY:-.venv/bin/python}"
NW="${NUM_WORKERS:-4}"
DEV="${DEVICE:-auto}"
export PYTHONPATH=.
step() { printf '\n== %s\n' "$*"; }

step "environment"
if [ ! -x "$PY" ]; then
  command -v uv >/dev/null || { echo "install uv (https://docs.astral.sh/uv/) or set PY=/path/to/python" >&2; exit 1; }
  uv venv --python 3.12 .venv
  uv pip install --python .venv/bin/python -r requirements.txt huggingface_hub
  PY=.venv/bin/python
fi
"$PY" -c "import torch, onnxruntime, huggingface_hub; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

DD="${DATA_DIR:-data}"
PK="${PACK_DIR:-data/packs}"
mkdir -p repro "$PK"
SM=(); [ "${SMOKE:-0}" = 1 ] && SM=(--max-train-clips 600 --max-test-clips 300 --epochs 2)

if [ "$MODE" = check ]; then
  step "download test split + synthetic negatives (Hugging Face)"
  "$PY" scripts/hf_to_dataset.py --download --hf $DD/hf --out $DD/me2-dataset --splits test
  step "pack"
  "$PY" scripts/pack_data.py --dataset $DD/me2-dataset --splits test --out "$PK"
  "$PY" scripts/pack_data.py --negatives $DD/me2-dataset/negatives --out "$PK"
  step "score the released ensemble (models/vcm_conformer_M_ens3.onnx, tau 0.40)"
  for s in test neg_test; do
    "$PY" -m vcm.eval --model models/vcm_conformer_M_ens3.onnx --pack "$PK" --split $s --tau 0.40 --final-test \
          --out repro/final/vcm_conformer_M_ens3_$s
  done
  "$PY" scripts/check_repro.py --repro repro/final --published results/final --names vcm_conformer_M_ens3_test \
        vcm_conformer_M_ens3_neg_test | tee repro/compare.md
  exit 0
fi

step "download all data (Hugging Face): gold splits + synthetic_negatives (train, test, noise, tune_oos)"
"$PY" scripts/hf_to_dataset.py --download --hf $DD/hf --out $DD/me2-dataset --noise-out $DD/noise

step "pack (train buffer 6 s, test / holdout 5 s windows, negatives, noise)"
NOISE=(); [ -n "$(ls $DD/noise/*.wav 2>/dev/null)" ] && NOISE=(--noise $DD/noise)
"$PY" scripts/pack_data.py --dataset $DD/me2-dataset --splits train test holdout --out "$PK" "${NOISE[@]}"
"$PY" scripts/pack_data.py --negatives $DD/me2-dataset/negatives --out "$PK"
[ -d $DD/me2-dataset/tune_oos ] && "$PY" scripts/pack_data.py --dataset $DD/me2-dataset --splits tune_oos --out "$PK"

step "tune split (15 % of train speakers, seed 0)"
"$PY" scripts/make_tune_split.py --pack-dir "$PK" --frac 0.15 --seed 0 --out "$PK/tune_split.json"

COMMON=(--select-split tune --use-negatives --misfire-weight 1.0 --num-workers "$NW" --device "$DEV" --pack-dir "$PK")
train() {  # arch tier seed outdir [extra...]
  local a=$1 t=$2 s=$3 o=$4; shift 4
  if [ -f "$o/summary.json" ]; then echo "done: $o"; return; fi
  local r=(); [ -f "$o/last.pt" ] && r=(--resume)
  "$PY" -m vcm.train --arch "$a" --tier "$t" --seed "$s" --out "$o" "${COMMON[@]}" "${SM[@]}" "$@" "${r[@]}"
}

step "final recipe: conformer M, seeds 0 1 2 (selection, early stopping on tune)"
for s in 0 1 2; do train conformer M $s repro/runs/conformer_M_s$s; done

if [ "$MODE" = full ]; then
  step "DS-CNN M baseline, same recipe"
  train ds_cnn M 0 repro/runs/ds_cnn_M_s0
  step "ablations on tune (conformer M seed 0)"
  B=(--select-split tune --num-workers "$NW" --device "$DEV" --pack-dir "$PK")
  for cfg in "base:" "mf1:--misfire-weight 1.0" "neg:--use-negatives" "neg_mf05:--use-negatives --misfire-weight 0.5" \
             "neg_mf2:--use-negatives --misfire-weight 2.0" "auglight:--use-negatives --misfire-weight 1.0 --aug light" \
             "augnone:--use-negatives --misfire-weight 1.0 --aug none"; do
    n=${cfg%%:*}; x=${cfg#*:}; o=repro/runs/ablation_$n
    [ -f "$o/summary.json" ] || "$PY" -m vcm.train --arch conformer --tier M --seed 0 --out "$o" "${B[@]}" "${SM[@]}" $x
  done
  step "bake-off rounds 1 and 2 (as published: selection on test)"
  PY="$PY" EXTRA="--num-workers $NW --device $DEV --pack-dir $PK" TAG=_repro scripts/bakeoff.sh
  PY="$PY" EXTRA="--num-workers $NW --device $DEV --pack-dir $PK" TAG=_repro scripts/bakeoff.sh results/round2_runs.txt
fi

step "export the 3-seed ensemble (tau 0.40) and int8"
mkdir -p repro/models
CK=repro/runs/conformer_M_s0/best.pt+repro/runs/conformer_M_s1/best.pt+repro/runs/conformer_M_s2/best.pt
"$PY" -m vcm.export export --ckpt "$CK" --out repro/models/vcm_conformer_M_ens3.onnx --tau 0.40 --pack "$PK"
"$PY" -m vcm.export quantize --onnx repro/models/vcm_conformer_M_ens3.onnx --out repro/models/vcm_conformer_M_ens3_int8.onnx --pack "$PK"
if [ -f "$PK/tune_oos_meta.parquet" ]; then
  step "reject threshold sweep on tune + tune_oos (published choice: 0.40)"
  "$PY" scripts/tau_report.py --model "$CK" --pack "$PK" --out repro/tau_ens3.csv || true
fi

step "score on test (final report)"
NAMES=()
for m in vcm_conformer_M_ens3 vcm_conformer_M_ens3_int8; do
  for s in test neg_test; do
    "$PY" -m vcm.eval --model repro/models/$m.onnx --pack "$PK" --split $s --tau 0.40 --final-test --out repro/final/${m}_$s
    NAMES+=("${m}_$s")
  done
done
if [ "$MODE" = full ]; then
  "$PY" -m vcm.eval --model repro/runs/ds_cnn_M_s0/best.pt --pack "$PK" --split test --tau 0.40 --final-test --out repro/final/ds_cnn_M_test
  NAMES+=(ds_cnn_M_test)
fi
"$PY" scripts/check_repro.py --repro repro/final --published results/final --names "${NAMES[@]}" | tee repro/compare.md
