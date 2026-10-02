set -u
export PY=/home/docquityadmin/vcm-env/bin/python
B="--num-workers 4 --select-split tune --use-negatives"
TAG=_tune_neg_mf05 EXTRA="$B --misfire-weight 0.5" bash scripts/bakeoff.sh results/misfire_runs.txt
TAG=_tune_neg_mf2  EXTRA="$B --misfire-weight 2.0" bash scripts/bakeoff.sh results/misfire_runs.txt
echo MISFIRE_CHAIN2_DONE
