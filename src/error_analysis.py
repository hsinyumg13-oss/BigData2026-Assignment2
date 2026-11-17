"""Error analysis of the LightGBM model on the validation weeks.

(1) precision / recall / F1 per activity (pooled over the weeks where the activity is recorded);
(2) recall split by whether the user did the same activity in the previous week (weeks 2 and 3,
    whose previous week is also a validation week) - how much of the error is 'new' activity.
Output: figures/error_analysis.csv, figures/error_analysis.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

d = pd.read_pickle("cache/valprob_lgb.pkl")
Ys, Ps, thr = d["Y"], d["P"], d["thr"]
rows = []
for k in thr:
    tp = fp = fn = 0; rec = {"repeat": [0, 0], "new": [0, 0]}
    for i, (Y, P) in enumerate(zip(Ys, Ps)):
        if k not in Y:
            continue
        y, p = Y[k].values.astype(bool), (P[k].values >= thr[k])
        tp += (y & p).sum(); fp += (~y & p).sum(); fn += (y & ~p).sum()
        if i > 0 and k in Ys[i - 1]:
            prev = Ys[i - 1][k].reindex(Y.index).fillna(0).values.astype(bool)
            for name, m in [("repeat", prev), ("new", ~prev)]:
                rec[name][0] += (y & p & m).sum(); rec[name][1] += (y & m).sum()
    pr, rc = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    r = {"activity": k, "precision": pr, "recall": rc, "f1": 2 * pr * rc / max(pr + rc, 1e-9), "positives": tp + fn}
    for name, (a, b) in rec.items():
        r[f"recall_{name}"] = a / b if b else np.nan
        r[f"share_{name}"] = b
    tot = r["share_repeat"] + r["share_new"]
    if tot:
        r["share_new"] = r["share_new"] / tot; r["share_repeat"] = r["share_repeat"] / tot
    rows.append(r)
df = pd.DataFrame(rows).round(3)
df.to_csv("figures/error_analysis.csv", index=False)
print(df.to_string())

fig, ax = plt.subplots(1, 2, figsize=(9, 3.0), dpi=200)
x = np.arange(len(df)); w = 0.27
for j, (c, col) in enumerate([("precision", "#9cc3e4"), ("recall", "#2a78b5"), ("f1", "#d1652f")]):
    ax[0].bar(x + (j - 1) * w, df[c], w, label=c, color=col)
ax[0].set_xticks(x, df.activity.str.replace("_", "\n"), fontsize=6.5); ax[0].set_ylim(0, 1)
ax[0].legend(fontsize=7, frameon=False, ncol=3); ax[0].set_title("LightGBM, pooled validation weeks", fontsize=8.5)
s = df.dropna(subset=["recall_repeat"])
x = np.arange(len(s))
ax[1].bar(x - 0.2, s.recall_repeat, 0.4, color="#3b9a5a", label="did it last week")
ax[1].bar(x + 0.2, s.recall_new, 0.4, color="#c4407a", label="did not do it last week")
for xi, sh in zip(x, s.share_new):
    ax[1].text(xi + 0.2, 0.02, f"{sh:.0%}\nof pos.", ha="center", fontsize=5.5, color="white")
ax[1].set_xticks(x, s.activity.str.replace("_", "\n"), fontsize=6.5); ax[1].set_ylim(0, 1)
ax[1].legend(fontsize=7, frameon=False); ax[1].set_title("recall by previous-week behaviour", fontsize=8.5)
for a in ax:
    for sp in ["top", "right"]:
        a.spines[sp].set_visible(False)
    a.tick_params(axis="y", labelsize=7)
fig.tight_layout(); fig.savefig("figures/error_analysis.png")
