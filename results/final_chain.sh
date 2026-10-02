set -u
export PY=/home/docquityadmin/vcm-env/bin/python
B="--num-workers 4 --select-split tune --use-negatives --misfire-weight 1.0"
TAG=_tune_neg_mf1        EXTRA="$B"              bash scripts/bakeoff.sh results/seeds12.txt
TAG=_tune_neg_mf1_augnone  EXTRA="$B --aug none"  bash scripts/bakeoff.sh results/misfire_runs.txt
TAG=_tune_neg_mf1_auglight EXTRA="$B --aug light" bash scripts/bakeoff.sh results/misfire_runs.txt
echo FINAL_CHAIN_DONE
