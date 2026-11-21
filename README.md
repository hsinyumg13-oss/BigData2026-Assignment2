# Assignment 2 – Predicting GitHub User Activity

**Big Data Analytics Techniques and Applications, Fall 2026 · Student ID 113701031**

For each of the 4,465 test users, predict whether they will **open_pr, open_issue, issue_comment,
pr_review, pr_review_comment, push** during **2026/11/22 00:00 – 11/29 00:00 (Taiwan time)**.

Two branches on the 8 relational tables, blended per activity:

| Branch | Model | Answers |
|---|---|---|
| Feature branch | temporal + foreign-key-join features per (user, cutoff t) → one LightGBM per activity | Q2 (a), Q3 |
| Graph branch (Relational Deep Learning) | every row a node, every FK an edge → heterogeneous GraphSAGE with a time encoding | Q2 (b), Q3 |

Report: [`slides/Asg2_113701031.pdf`](slides/Asg2_113701031.pdf)

## Data Source

The data is sourced from **[GH Archive](https://www.gharchive.org/)** (https://www.gharchive.org/), a public
archive of GitHub events, and processed with the course script `scripts/preprocess.py` (unchanged), which
builds the 8 tables and anonymizes them. No other data is used (no GitHub API, no other GH Archive event
types, no other websites). For educational use within the course.

## How to run

```bash
pip install -r requirements.txt     # CPU torch: add --extra-index-url https://download.pytorch.org/whl/cpu
./run_all.sh                        # -> output_113701031.csv + official format check
```

`run_all.sh` runs 7 steps; each script can also be run on its own (the exact commands are in `run_all.sh`).

| Step | Script | Output |
|---|---|---|
| 1. Download GH Archive, keep the cohort's events | `src/download_filter.py` | `gharchive_cohort/*.json.gz` |
| 2. Build the 8 tables (course script, unchanged) | `scripts/preprocess.py` | `data/db/*.parquet` |
| 3. Daily coverage / capture rate per event type | `src/coverage.py` | `data/coverage.csv` |
| 4. Feature branch: snapshots, validation, LightGBM × 6 | `src/train.py` | probabilities in `cache/` |
| 5. Graph branch (RDL): GraphSAGE | `src/gnn.py` | probabilities in `cache/` |
| 6. Blend + thresholds → final predictions | `src/blend.py` | `output_113701031.csv` |
| 7. Format check | `scripts/evaluate.py` | – |

Notes

* **Step 1** downloads each hour file and keeps only the lines `preprocess.py` itself keeps (it calls
  `preprocess.extract_hour`: cohort users, the 6 event types, no bots), so step 2 gives the same tables as on
  the raw files with ~1 % of the disk space. The download is resumable; an hour that GH Archive answers with
  404 is marked `*.missing` and skipped. If you already have the raw files, run step 2 on your own folder.
* **Reproducible:** the data end is fixed (`DATA_END` in `run_all.sh`), so a later run downloads the same
  8,532 hours and ignores anything after it. LightGBM runs in deterministic mode, the GNN with a fixed seed,
  and package versions are pinned in `requirements.txt`.

## Dataset time period

| | UTC | Taiwan time |
|---|---|---|
| GH Archive hours used (8,532 files) | 2025-12-01 00:00 – 2026-11-21 12:00 (excl.) | 2025-12-01 08:00 – 2026-11-21 20:00 |
| Training label weeks (36) | cutoffs t = 2026-02-14 12:00 … 10-17 12:00, every 7 days | t at 20:00 |
| Validation label weeks (3) | cutoffs 2026-10-24, 10-31, 11-07 12:00 | t at 20:00 |
| Test week | 2026-11-21 16:00 – 11-28 16:00 | 2026-11-22 00:00 – 11-29 00:00 |

Each label week is [t + 4 h, t + 4 h + 7 d): the same 4-hour gap as between the end of the data and the test
week. The **Results** below come from experiments with data end 2026-11-16 00:00 UTC, so their validation
cutoffs are 2026-10-19 / 10-26 / 11-02 (5.5 days earlier, same gap).

## Method

* **Point-in-time snapshots.** One sample per (user, weekly cutoff t). A row is used only if its row time
  (`first_seen_at` for entity tables, `created_at` for comments / pushes) is before t; "known later" columns
  (`merged_at`, `merged_by_id`, `closed_at`, `state_reason`, `dismissed_at`) only if their own time is before
  t. Users: ≥ 3 target activities in the 91 days before t (reproduces `test_input.csv`, F1 0.996).
* **GH Archive capture.** Since June 2026 GH Archive records only 10–30 % of the events (push ~0–3 % since
  10/27, `figures/capture_rate.png`). A (week, activity) label is used only if ≥ 70 % of the week was
  recorded, and recency counts only recorded days.
* **Q2 – relations across tables.** (a) FK-join features: own PRs / issues, reviews and comments received,
  activity of others in the user's repositories, review demand, ownership, push details, merges.
  (b) RDL: rows with row time in [t − 56 d, t) as nodes, FKs (+ reverse) as edges, 2-layer heterogeneous
  GraphSAGE, user nodes → 6 logits.
* **Q3 – time.** Counts in 1–112-day windows, 8 weekly lags, exponential decay (half-lives 3 / 14 / 60 d),
  recency, regularity, weekday / night share, trend; in the GNN a time encoding of t − row time.
* **Selection.** Per activity, the threshold (max F1) and the blend weight
  p = w · p_LightGBM + (1 − w) · p_GNN are chosen on the validation weeks only; both branches are then
  retrained on all weeks and predict the test week.

## Results

Score = (macro-F1 + micro-F1) / 2.

| Model | Validation (in-sample) | Leave-one-week-out | Back-test 2026/09/27 |
|---|---|---|---|
| copy last week | 0.3732 | – | 0.4994 (TA: 0.4961) |
| TA README: GNN without / with time | – | – | 0.3570 / 0.5482 |
| LightGBM (T + R) | 0.4481 | 0.4333 | 0.5317 |
| GNN without time encoding | 0.4185 | – | 0.4847 |
| GNN + time encoding (RDL) | 0.4491 | 0.4421 | 0.5283 |
| **hybrid (submitted)** | 0.4599 | 0.4333 | **0.5399** |

* On the unseen back-test week the hybrid beats LightGBM by +0.008 (95 % paired-bootstrap CI 0.003 – 0.013),
  so it is submitted; in leave-one-week-out the in-sample gain of +0.012 disappears.
* The time encoding is decisive for the GNN: +0.031 on validation, +0.044 on the back-test.
* Main error: 58–76 % of the positives did not do that activity the week before; recall on them is
  17–30 %, against 70–90 % for repeat users.
* Limitation: only 3 validation weeks (push recorded in one), and the test labels come from the same
  under-recording archive.

<details>
<summary>Q2 ablation (LightGBM, base T + R = 0.4481; T = temporal, R = relational, C = capture-rate)</summary>

| Variant | Validation | Difference [95 % CI] |
|---|---|---|
| temporal only (no relational) | 0.4503 | +0.0021 [-0.0039, +0.0087] |
| T + R + C | 0.4226 | – |
| T + C | 0.4157 | -0.0325 [-0.0451, -0.0208] |
| − own PRs / issues | 0.4492 | +0.0010 [-0.0051, +0.0067] |
| − feedback received | 0.4499 | +0.0017 [-0.0049, +0.0080] |
| − repository neighbourhood | 0.4501 | +0.0019 [-0.0038, +0.0070] |
| − review demand | 0.4516 | +0.0034 [-0.0025, +0.0096] |
| − ownership / organisation | 0.4424 | -0.0057 [-0.0127, +0.0007] |
| − push details | 0.4484 | +0.0002 [-0.0062, +0.0067] |
| RDL GNN (no hand-made features) | 0.4491 | +0.0010 [-0.0081, +0.0117] |

No relational group changes the score beyond noise. C hurts on validation (and on the weeks before 9/27:
T + R + C 0.5060 vs T + R 0.5488) but helps on 9/27 (0.5459 vs 0.5317): unstable, so left out.
</details>

<details>
<summary>Q3 ablation (difference to T + R)</summary>

| Variant | Validation | Difference [95 % CI] |
|---|---|---|
| − window counts | 0.4471 | -0.0011 [-0.0070, +0.0046] |
| − weekly lags | 0.4517 | +0.0035 [-0.0024, +0.0097] |
| − exponential decay | 0.4463 | -0.0018 [-0.0080, +0.0040] |
| − recency | 0.4461 | -0.0021 [-0.0089, +0.0047] |
| − regularity | 0.4501 | +0.0019 [-0.0037, +0.0075] |
| − habits / trend | 0.4480 | -0.0001 [-0.0072, +0.0074] |
| only plain window counts | 0.4522 | +0.0040 [-0.0056, +0.0137] |
| recency in calendar days | 0.4479 | -0.0002 [-0.0066, +0.0075] |
| unrecorded weeks as negatives | 0.4525 | +0.0044 [-0.0004, +0.0095] |
| recency-weighted training (half-life 8 w) | 0.4397 | -0.0084 [-0.0156, -0.0010] |
| only 12 training weeks | 0.4396 | -0.0085 [-0.0181, +0.0004] |
| GNN without time encoding | 0.4185 | -0.0297 [-0.0398, -0.0187] |
| GNN with time encoding | 0.4491 | +0.0010 [-0.0081, +0.0117] |

Significant: the GNN time encoding (helps) and recency-weighted training (hurts). The temporal groups
overlap, so removing one alone does not change the score.
</details>

To reproduce every number: `./run_experiments.sh all`, then `python src/significance.py`,
`python src/bt_bootstrap.py`, `python src/results_figures.py`, `python src/error_analysis.py`.

## Repository layout

```
run_all.sh            full pipeline (steps 1–7)
run_experiments.sh    all experiments of the report
scripts/              course files: preprocess.py, evaluate.py, cohort.txt (unchanged)
submission/           test_input.csv, sample_submission.csv (unchanged)
src/
  download_filter.py  step 1 – download + cohort filter
  coverage.py         step 3 – daily coverage / capture rate
  features.py         temporal and relational features, labels, cut at the data end
  train.py            step 4 – snapshots, validation, LightGBM, thresholds
  gnn.py              step 5 – relational entity graph + heterogeneous GraphSAGE
  blend.py            step 6 – blend of the two branches
  significance.py, bt_bootstrap.py, error_analysis.py    statistics for the report
  *_figure.py, schema_graph.py, results_figures.py       figures
figures/              schema graph, timeline, framework, capture rate, ablations, error analysis
slides/               report (pptx / pdf) and build_slides.py
```
