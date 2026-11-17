"""Detect days on which GH Archive is missing an event type.

GH Archive itself has gaps: some event types are missing on whole days (e.g. no PushEvent
since late October), and since June 2026 the non-push types are only partly recorded.
We count, per day and per event type, the cohort's events in the GH Archive hour files and
divide by the median daily count of a stable reference period (Dec 2025 - May 2026):
  capture = how much of the normal volume GH Archive recorded that day (continuous)
  covered = capture >= COV_RATIO (the type was recorded at all that day)
Used to
  * mask labels of weeks whose target activity was not recorded (not real negatives), and
  * measure recency / window counts in COVERED days only (features.py, covered age).
"""
import gzip
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

TYPE2KIND = {"PushEvent": "push", "PullRequestEvent": "open_pr", "IssuesEvent": "open_issue",
             "IssueCommentEvent": "issue_comment", "PullRequestReviewEvent": "pr_review",
             "PullRequestReviewCommentEvent": "pr_review_comment"}
COV_RATIO = 0.02
REF_START, REF_END = "2025-12-01", "2026-05-31"


def daily_counts(raw_dir):
    cnt = defaultdict(Counter)
    for f in sorted(Path(raw_dir).glob("*.json.gz")):
        m = re.match(r"(\d{4}-\d{2}-\d{2})-(\d+)\.json\.gz$", f.name)
        if not m:
            continue
        day = m.group(1)
        cnt[day]["_hours"] += 1
        for ln in gzip.open(f):
            t = json.loads(ln)["type"]
            if t in TYPE2KIND:
                cnt[day][TYPE2KIND[t]] += 1
    df = pd.DataFrame(cnt).T.fillna(0).sort_index()
    df.index = pd.to_datetime(df.index)
    return df


def coverage(df):
    kinds = list(TYPE2KIND.values())
    per_hour = df[kinds].div(df["_hours"].clip(lower=1), axis=0) * 24
    ref = per_hour.loc[REF_START:REF_END].median()
    ratio = (per_hour / ref).clip(upper=2.0)
    cov = (ratio >= COV_RATIO).astype(float) * (df["_hours"] / 24).clip(upper=1).values[:, None]
    cov["merge_pr"] = cov["open_pr"]  # merges come from PullRequestEvent
    ratio["merge_pr"] = ratio["open_pr"]
    return cov, ratio


if __name__ == "__main__":
    raw, out = sys.argv[1], sys.argv[2]
    df = daily_counts(raw)
    cov, ratio = coverage(df)
    df.to_csv(out.replace(".csv", "_counts.csv"))
    cov.to_csv(out)
    ratio.to_csv(out.replace(".csv", "_capture.csv"))
    print("weekly mean capture (share of the Dec-May volume):")
    print(ratio.resample("W").mean().round(3).to_string())
    bad = cov[(cov < 0.5).any(axis=1)]
    print(f"{len(df)} days; days with a missing type:")
    print(bad.round(2).to_string())
