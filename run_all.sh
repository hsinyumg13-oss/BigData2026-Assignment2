#!/bin/bash
# Full pipeline: GH Archive -> 8 tables (course preprocess.py) -> features -> LightGBM -> output CSV
# Usage: ./run_all.sh
#
# DATA_START / DATA_END fix the exact GH Archive hours used (UTC, end exclusive), so a re-run
# at any later time downloads the same files and reproduces output_113701031.csv.
set -e
STUDENT_ID=113701031
DATA_START="2025-12-01 00:00"     # UTC (= 2025-12-01 08:00 Taiwan); preprocess.py ignores earlier files
DATA_END="2026-11-21 12:00"       # UTC (= 2026-11-21 20:00 Taiwan), exclusive
TEST_START="2026-11-22 00:00"     # Taiwan time, start of the test week

# 1. download the GH Archive hour files of [DATA_START, DATA_END) and keep only the cohort's
#    events, with the same filter as preprocess.py (resumable)
python src/download_filter.py --start "$DATA_START" --end "$DATA_END" --out gharchive_cohort --workers 8

# 2. build the 8 relational tables with the course script (unchanged)
python scripts/preprocess.py --raw gharchive_cohort --out data

# 3. daily coverage / capture rate of every event type in GH Archive
python src/coverage.py gharchive_cohort data/coverage.csv

# 4. feature branch: features, time-split validation, LightGBM x 6 (probabilities -> cache/)
python src/train.py --db data/db --coverage data/coverage.csv --data_end "$DATA_END" \
    --target_start "$TEST_START" --out output_lgb.csv --n_weeks 40 --n_val 3 --tag lgb

# 5. graph branch (RDL): heterogeneous GraphSAGE on the relational entity graph
python src/gnn.py --db data/db --coverage data/coverage.csv --data_end "$DATA_END" \
    --target_start "$TEST_START" --n_weeks 40 --n_val 3 --tag gnn_time

# 6. hybrid (weights and thresholds from the validation weeks) -> final output
python src/blend.py --lgb lgb --gnn gnn_time --out output_${STUDENT_ID}.csv

# 7. format check
python scripts/evaluate.py --pred output_${STUDENT_ID}.csv --input submission/test_input.csv
