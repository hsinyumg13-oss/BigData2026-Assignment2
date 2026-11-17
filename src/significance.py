"""Paired bootstrap over users: is an ablation's validation Score really different from the base?

For every tag with cache/valprob_<tag>.pkl we take its binary predictions (its own thresholds),
resample the users of each validation week with replacement (same resample for both runs),
and report the mean-over-weeks Score difference to the base with a 95% interval.
Output: figures/significance.csv
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def f1(y, p):
    tp = (y & p).sum(0); fp = (~y & p).sum(0); fn = (y & ~p).sum(0)
    return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0), tp, fp, fn


def week_score(y, p):
    """y, p: bool arrays (n_users, n_act) -> (macro + micro) / 2"""
    f, tp, fp, fn = f1(y, p)
    d = 2 * tp.sum() + fp.sum() + fn.sum()
    micro = 2 * tp.sum() / d if d else 0.0
    return (f.mean() + micro) / 2


def load(tag, cache):
    d = pd.read_pickle(Path(cache) / f"valprob_{tag}.pkl")
    out = []
    for Y, P in zip(d["Y"], d["P"]):
        cols = list(Y.columns)
        B = pd.DataFrame({k: P[k] >= d["thr"][k] for k in cols}, index=P.index)
        out.append((Y[cols].astype(bool), B[cols]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="lgb")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--n_boot", type=int, default=1000)
    a = ap.parse_args()
    base = load(a.base, a.cache)
    rng = np.random.default_rng(0)
    tags = sorted(Path(p).stem[len("valprob_"):] for p in glob.glob(f"{a.cache}/valprob_*.pkl"))
    rows = []
    for t in tags:
        if t == a.base:
            continue
        other = load(t, a.cache)
        diffs = np.zeros(a.n_boot); point = 0.0
        for w, ((Yb, Bb), (Yo, Bo)) in enumerate(zip(base, other)):
            u = Yb.index.intersection(Yo.index)
            cols = [c for c in Yb.columns if c in Yo.columns]
            yb, pb = Yb.loc[u, cols].values, Bb.loc[u, cols].values
            yo, po = Yo.loc[u, cols].values, Bo.loc[u, cols].values
            point += (week_score(yo, po) - week_score(yb, pb)) / len(base)
            n = len(u)
            for b in range(a.n_boot):
                i = rng.integers(0, n, n)
                diffs[b] += (week_score(yo[i], po[i]) - week_score(yb[i], pb[i])) / len(base)
        lo, hi = np.quantile(diffs, [0.025, 0.975])
        rows.append({"tag": t, "diff": round(point, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                     "significant": bool(lo > 0 or hi < 0)})
        print(rows[-1])
    pd.DataFrame(rows).to_csv("figures/significance.csv", index=False)


if __name__ == "__main__":
    main()
