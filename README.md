# BigData 2026 Fall – Assignment 2: Predicting GitHub User Activity

Student ID: 113701031

For every test user we predict whether they will perform each of 6 activities
(`open_pr, open_issue, issue_comment, pr_review, pr_review_comment, push`) during the test week
**2026/11/22 00:00 – 11/29 00:00 (Taiwan time)**. The model reads the 8 relational tables built by
the course script `scripts/preprocess.py` and combines two branches:

* **Feature branch** – temporal and cross-table (foreign-key) features per (user, cutoff time),
  one LightGBM classifier per activity;
* **Graph branch (Relational Deep Learning)** – every row of every table is a node and every
  foreign key an edge; a heterogeneous GraphSAGE with a time encoding predicts the 6 activities
  from the Users nodes;

and blends them per activity (weights and thresholds chosen on the validation weeks only).

## Data Source

The data used in this assignment is sourced from **[GH Archive](https://www.gharchive.org/)**
(https://www.gharchive.org/), a publicly available archive of GitHub events. It is processed with
the course script `scripts/preprocess.py` (unchanged), which builds the 8 tables and anonymizes them.
No other data is used (no GitHub API, no other GH Archive event types, no other websites). This
work is for educational use within the course.

## Dataset time period

| | UTC | Taiwan time |
|---|---|---|
| GH Archive hour files used | 2025-12-01 00:00 – 2026-11-21 12:00 (exclusive), 8,532 files | 2025-12-01 08:00 – 2026-11-21 20:00 |
| Training label weeks | 36 weeks, cutoffs t = 2026-02-14 12:00 … 2026-10-17 12:00 (every 7 days); labels in [t + 4 h, t + 4 h + 7 d) | cutoffs at 20:00 |
| Validation label weeks | 3 weeks, cutoffs 2026-10-24, 10-31, 11-07 12:00 (rolling time split, all after the training weeks) | cutoffs at 20:00 |
| Test week | 2026-11-21 16:00 – 2026-11-28 16:00 | 2026-11-22 00:00 – 11-29 00:00 |

The end of the data is **fixed** (`DATA_END` in `run_all.sh`, `--data_end` in `src/train.py`):
the scripts download exactly these hours and ignore any row after `DATA_END`, so running the
pipeline at any later time reproduces `output_113701031.csv`. LightGBM runs in deterministic mode
(fixed seed, 2 threads); the GNN uses a fixed seed.

## How to run

```bash
pip install -r requirements.txt   # CPU torch: add --extra-index-url https://download.pytorch.org/whl/cpu
./run_all.sh            # -> output_113701031.csv, then the official format check
```

`run_all.sh` runs these steps (each can be run on its own):

| Step | Command | Output |
|---|---|---|
| 1. Download GH Archive, keep the cohort's events | `python src/download_filter.py --start "2025-12-01 00:00" --end "2026-11-21 12:00" --out gharchive_cohort` | `gharchive_cohort/YYYY-MM-DD-H.json.gz` |
| 2. Build the 8 tables (course script) | `python scripts/preprocess.py --raw gharchive_cohort --out data` | `data/db/*.parquet` |
| 3. Coverage / capture rate per event type | `python src/coverage.py gharchive_cohort data/coverage.csv` | `data/coverage*.csv` |
| 4. Feature branch: validation, training, prediction | `python src/train.py --db data/db --coverage data/coverage.csv --data_end "2026-11-21 12:00" --target_start "2026-11-22 00:00" --out output_lgb.csv` | probabilities in `cache/` |
| 5. Graph branch (RDL) | `python src/gnn.py --db data/db --coverage data/coverage.csv --data_end "2026-11-21 12:00" --target_start "2026-11-22 00:00" --tag gnn_time` | probabilities in `cache/` |
| 6. Hybrid + final output | `python src/blend.py --lgb lgb --gnn gnn_time --out output_113701031.csv` | `output_113701031.csv` |
| 7. Format check | `python scripts/evaluate.py --pred output_113701031.csv --input submission/test_input.csv` | |

**Step 1 note.** A raw GH Archive hour file is 20–200 MB. `download_filter.py` downloads each
hour, keeps only the lines that `preprocess.py` itself keeps (it calls `preprocess.extract_hour`:
actor in the 9,000-user cohort, the 6 event types, no bots), writes them back under the same file
name and deletes the raw file. `preprocess.py` on this folder gives the same tables as on the raw
files. If you already have the raw files, run step 2 on your own folder instead; steps 4–6 ignore
everything after `DATA_END`.

## Repository layout

```
scripts/                 course files (preprocess.py, evaluate.py, cohort.txt) – unchanged
submission/              test_input.csv, sample_submission.csv – unchanged
src/
  download_filter.py     step 1
  coverage.py            step 3: daily coverage + capture rate of each event type
  features.py            temporal and relational features, labels, cut at DATA_END
  train.py               snapshots, time-split validation, LightGBM x 6, thresholds
  gnn.py                 relational entity graph + heterogeneous GraphSAGE (RDL)
  blend.py               hybrid of the two branches
  significance.py        paired bootstrap of the ablations (validation weeks)
  bt_bootstrap.py        paired bootstrap on the back-test week
  error_analysis.py      precision / recall, repeat vs new activity
  schema_graph.py  timeline_figure.py  framework_figure.py  capture_figure.py  results_figures.py   figures
run_all.sh               full pipeline
run_experiments.sh       all experiments of the report
slides/                  Asg2_113701031.pptx / .pdf and build_slides.py (fills the TA template)
figures/                 schema_graph.png, timeline.png, framework.png, capture_rate.png, ...
```

## Method

**Snapshots.** One training sample per (user, weekly cutoff t). A row of a table may be used at t only if
its row time (`first_seen_at` for Users / Repositories / PullRequests / Issues / PRReviews, `created_at`
for IssueComments / PRReviewComments / Pushes) is < t; the "known later" columns (`merged_at`,
`merged_by_id`, `closed_at`, `state_reason`, `dismissed_at`) are used only if their own time is < t
(`features.cut_tables`). Labels: whether the user does each activity in [t + 4 h, t + 4 h + 7 d) – the same
4-hour gap as between the end of our data and the test week. Users: >= 3 target activities in the 91 days
before t (this rule reproduces `test_input.csv` with F1 0.996).

**GH Archive capture.** Since June 2026 GH Archive records only 10–30 % of the cohort's events
(PushEvents ~0–3 % since 10/27, see `figures/capture_rate.png`). `src/coverage.py` measures it per day and
event type; a (week, activity) label is used only if >= 70 % of the week was recorded (otherwise it is
masked in the loss and in validation), and recency features count only recorded days.

**Q2 – relations across tables.** (a) Feature branch: FK-join features – the user's own PRs / issues
(open, merged in 28 d, draft / cross-repo), reviews and comments *received* from others
(PRReviews / PRReviewComments -> PullRequests -> author, IssueComments -> Issues -> author), activity of
other users in the user's repositories, review demand (new PRs in repos the user reviews), ownership
(Users.login = Repositories.owner_login, org repos), push details, merges (`merged_by_id`, usable from
`merged_at`). (b) Graph branch (Relational Deep Learning, `src/gnn.py`): every row with row time in
[t – 56 d, t) is a node, every FK an edge (+ reverse edge); node inputs are the row's columns plus a time
encoding; heterogeneous GraphSAGE (2 layers, hidden 64, mean + sum aggregation, residual + LayerNorm), user
nodes -> 6 logits, masked BCE.

**Q3 – time.** Counts in 1/3/7/14/28/56/112-day windows, 8 weekly lags, exponential decay (half-lives
3/14/60 d), days since last / first activity (in recorded days), share of active weeks (4/8/16), weekday /
night share, trend; in the GNN: log age, decays (1/7/28 d) and sin/cos of (t – row time). Rolling time
split: 36 training weeks, then 3 validation weeks.

**Model selection.** One LightGBM per activity (600 rounds, lr 0.03, 31 leaves, deterministic). Per
activity, the threshold (max F1) and the hybrid weight p = w·p_LGB + (1–w)·p_GNN are chosen on the 3
validation weeks; then both branches are retrained on all weeks and predict the test week.

## Results

Score = (macro-F1 + micro-F1) / 2. Validation = 3 weeks with cutoffs 2026-10-19 / 10-26 / 11-02 (data end
2026-11-16 for these experiments; push is recorded only in the first week). `[ ]` = 95 % paired-bootstrap
interval over users of the difference to the base model (`src/significance.py`).

| Model | Validation (in-sample thresholds) | Leave-one-week-out | Back-test week 2026/09/27 |
|---|---|---|---|
| copy last week | 0.3732 | – | 0.4994 (TA README: 0.4961) |
| TA README: GNN without / with time | – | – | 0.3570 / 0.5482 |
| LightGBM (T + R) | 0.4481 | 0.4333 | 0.5317 |
| GNN without time encoding | 0.4185 | – | 0.4847 |
| GNN + time encoding (RDL) | 0.4491 | 0.4421 | 0.5283 |
| **hybrid (submitted)** | 0.4599 | 0.4333 | **0.5399** |

* In-sample, the hybrid looks +0.012 better than LightGBM; with weights and thresholds tuned on the other
  two weeks (leave-one-week-out) the gain disappears. On the back-test week – whose weights / thresholds come
  only from the 3 weeks before it – the hybrid beats LightGBM by +0.008
  (95 % CI 0.003 – 0.013). We submit the hybrid.
* The time encoding is decisive for the GNN: +0.031 on validation,
  +0.044 on the back-test.
* On 9/27 we stay ~0.01 below the TA's GNN + time reference, with a comparable copy-last-week baseline.

### Q2 ablation (LightGBM, base = T + R = 0.4481)

| Variant | Validation | Difference [95 % CI] |
|---|---|---|
| temporal only (no relational) | 0.4503 | +0.0021 [-0.0039, +0.0087] |
| T + R + capture-rate features | 0.4226 | - |
| T + capture-rate features | 0.4157 | -0.0325 [-0.0451, -0.0208] |
| - own PRs / issues | 0.4492 | +0.0010 [-0.0051, +0.0067] |
| - feedback received | 0.4499 | +0.0017 [-0.0049, +0.0080] |
| - repository neighbourhood | 0.4501 | +0.0019 [-0.0038, +0.0070] |
| - review demand | 0.4516 | +0.0034 [-0.0025, +0.0096] |
| - ownership / organisation | 0.4424 | -0.0057 [-0.0127, +0.0007] |
| - push details | 0.4484 | +0.0002 [-0.0062, +0.0067] |
| RDL GNN (no hand-made features) | 0.4491 | +0.0010 [-0.0081, +0.0117] |

No single relational group changes the score beyond noise; the capture-rate features (C) hurt
significantly. Back-test check of the same choices (LightGBM, week 2026/09/27): T + R 0.5317, T only 0.5361,
T + C + R 0.5459 – but on the three validation weeks before 9/27, T + C + R is 0.5060 against 0.5488 for
T + R. C is unstable across weeks (it can only shift the weekly base rate), so it is left out, as chosen on
validation. The GNN reaches the LightGBM level from the raw rows alone.

### Q3 ablation

| Variant | Validation | Difference to T + R [95 % CI] |
|---|---|---|
| - window counts | 0.4471 | -0.0011 [-0.0070, +0.0046] |
| - weekly lags | 0.4517 | +0.0035 [-0.0024, +0.0097] |
| - exponential decay | 0.4463 | -0.0018 [-0.0080, +0.0040] |
| - recency | 0.4461 | -0.0021 [-0.0089, +0.0047] |
| - regularity | 0.4501 | +0.0019 [-0.0037, +0.0075] |
| - habits / trend | 0.4480 | -0.0001 [-0.0072, +0.0074] |
| only plain window counts | 0.4522 | +0.0040 [-0.0056, +0.0137] |
| recency in calendar days | 0.4479 | -0.0002 [-0.0066, +0.0075] |
| unrecorded weeks as negatives | 0.4525 | +0.0044 [-0.0004, +0.0095] |
| recency-weighted training (half-life 8 w) | 0.4397 | -0.0084 [-0.0156, -0.0010] |
| only 12 training weeks | 0.4396 | -0.0085 [-0.0181, +0.0004] |
| GNN without time encoding | 0.4185 | -0.0297 [-0.0398, -0.0187] |
| GNN with time encoding | 0.4491 | +0.0010 [-0.0081, +0.0117] |

Significant: the GNN time encoding, and recency-weighted training (hurts). The temporal feature groups
overlap, so removing one alone does not change the score.

### Error analysis (`src/error_analysis.py`)

58–76 % of the positives are users who did not do that activity in the previous week; recall on them is
17–30 %, against 70–90 % for repeat users. With 10–30 % capture, part of "new" activity is only
"not recorded last week". Push is predicted best (F1 0.68), pr_review_comment worst (0.33).

### Limitations

Only 3 validation weeks (push in one); the test labels come from the same under-recording archive, so the
test score depends on GH Archive's capture during the test week.

All numbers: `./run_experiments.sh all`, then `python src/significance.py`, `python src/bt_bootstrap.py`,
`python src/results_figures.py`, `python src/error_analysis.py`.

