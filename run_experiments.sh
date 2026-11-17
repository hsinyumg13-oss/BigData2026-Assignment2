#!/bin/bash
# Experiments for the report (Q2-Q4). Validation only: rolling time split, last 3 weeks.
# Base model = LightGBM on temporal (T) + relational (R) features. Resumable: a run whose
# cache/exp_<tag>.json exists is skipped.  Usage: ./run_experiments.sh [part] [END]
#   part: q2 | q3 | gnn | backtest | all
set -e
PART=${1:-all}
END=${2:-"2026-11-16T00:00"}
# same 4-hour gap as the final run (data ends 20:00 Taiwan, test week starts 00:00 Taiwan)
TGT=$(python -c "import pandas as pd,sys; print((pd.Timestamp(sys.argv[1])+pd.Timedelta(hours=12)).strftime('%Y-%m-%dT%H:%M'))" "$END")
A="--db data/db --coverage data/coverage.csv --data_end $END --target_start $TGT --out /dev/null --n_weeks 40 --n_val 3 --no_final"
G="--db data/db --coverage data/coverage.csv --data_end $END --target_start $TGT --n_weeks 40 --n_val 3"
mkdir -p logs
run() { tag=$1; shift; [ -f cache/exp_$tag.json ] && { echo "== $tag (done)"; return; }
        echo "== $tag"; python src/train.py $A --tag $tag "$@" > logs/exp_$tag.log 2>&1; grep "MEAN VAL" logs/exp_$tag.log; }
gnn() { tag=$1; shift; [ -f cache/exp_$tag.json ] && { echo "== $tag (done)"; return; }
        echo "== $tag"; python src/gnn.py "$@" --tag $tag > logs/exp_$tag.log 2>&1; grep "MEAN VAL\|BACKTEST" logs/exp_$tag.log; }

if [[ $PART == q2 || $PART == all ]]; then
  run lgb      --groups T,R                    # base model
  run T_only   --groups T                      # no relational features
  run T_C_R    --groups T,C,R                  # + capture-rate features
  run T_C      --groups T,C
  for g in R_own R_recv R_nb R_demand R_owner R_push; do run no_$g --groups T,R --drop $g; done
fi
if [[ $PART == q3 || $PART == all ]]; then
  for g in T_win T_lag T_ewm T_rec T_reg T_hab; do run no_$g --groups T,R --drop $g; done
  run basic_counts --groups T --basic
  run raw_age  --groups T,R --raw_age
  run no_mask  --groups T,R --no_mask
  run hl8      --groups T,R --halflife 8
  run weeks12  --groups T,R --n_weeks 12
fi
if [[ $PART == gnn || $PART == all ]]; then
  gnn gnn_time   $G
  gnn gnn_notime $G --no_time
  python src/blend.py --lgb lgb --gnn gnn_time | tee logs/exp_hybrid.log
fi
if [[ $PART == backtest || $PART == all ]]; then
  # back-test on the README reference week 2026/09/27 - 10/04 (compare with the TA baselines)
  B="--db data/db --coverage data/coverage.csv --data_end 2026-09-26T16:00 --target_start 2026-09-27T00:00 --n_weeks 40 --n_val 3 --score_final"
  [ -f cache/btprob_bt_lgb.pkl ] || { python src/train.py $B --out /dev/null --tag bt_lgb > logs/bt_lgb.log 2>&1; }
  grep BACKTEST logs/bt_lgb.log
  gnn bt_gnn $B
  gnn bt_gnn_nt $B --no_time
  python src/blend.py --lgb bt_lgb --gnn bt_gnn --backtest | tee logs/bt_hybrid.log
fi
