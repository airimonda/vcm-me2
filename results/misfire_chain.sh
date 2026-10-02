set -u
export PY=/home/docquityadmin/vcm-env/bin/python
B="--num-workers 4 --select-split tune"
TAG=_tune_base     EXTRA="$B"                                         bash scripts/bakeoff.sh results/misfire_runs.txt
TAG=_tune_mf1      EXTRA="$B --misfire-weight 1.0"                    bash scripts/bakeoff.sh results/misfire_runs.txt
TAG=_tune_neg      EXTRA="$B --use-negatives"                         bash scripts/bakeoff.sh results/misfire_runs.txt
TAG=_tune_neg_mf1  EXTRA="$B --use-negatives --misfire-weight 1.0"    bash scripts/bakeoff.sh results/misfire_runs.txt
echo MISFIRE_CHAIN_DONE
