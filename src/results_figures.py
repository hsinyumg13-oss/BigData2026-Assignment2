"""Report figures from the experiment results in cache/exp_*.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

C = Path("cache")


def load(tag):
    p = C / f"exp_{tag}.json"
    return json.load(open(p)) if p.exists() else None


def bars(rows, title, out, base_tag="lgb", copy=True):
    """rows: [(label, tag)]. Horizontal bars of mean validation Score; full model highlighted."""
    full = load(base_tag)
    sig = pd.read_csv("figures/significance.csv").set_index("tag") if Path("figures/significance.csv").exists() else None
    data = [(lab, t, load(t)) for lab, t in rows]
    data = [(lab, t, r["mean_model"]) for lab, t, r in data if r]
    if copy and full:
        data.append(("copy last week (baseline)", None, full["mean_copy"]))
    if not data:
        return
    labs, tags, vals = zip(*data)
    fig, ax = plt.subplots(figsize=(7.6, 0.36 * len(data) + 0.9), dpi=200)
    colors = ["#2a78b5" if l.startswith("full") else ("#999999" if "baseline" in l else "#9cc3e4") for l in labs]
    ax.barh(range(len(vals)), vals, color=colors)
    for i, v in enumerate(vals):
        d = f"  {v:.4f}" + (f"  ({v - full['mean_model']:+.4f})" if full and not labs[i].startswith("full")
                             and "baseline" not in labs[i] else "")
        x = v
        if sig is not None and tags[i] in sig.index:      # 95% bootstrap interval of the difference
            r = sig.loc[tags[i]]
            b = full["mean_model"]
            ax.plot([b + r.ci_low, b + r.ci_high], [i, i], color="#333", lw=1)
            x = max(v, b + r.ci_high)
            d += " *" if r.significant else ""
        ax.text(x, i, d, va="center", fontsize=7.5)
    ax.set_yticks(range(len(vals)), labs, fontsize=8)
    ax.invert_yaxis()
    lo = min(vals) - 0.03
    ax.set_xlim(lo, max(vals) + 0.075)
    ax.set_xlabel("mean validation Score = (macro-F1 + micro-F1) / 2 over 3 weeks\nline = 95% paired-bootstrap interval of the "
                  "difference to the full model,  * = interval excludes 0", fontsize=7.5)
    ax.set_title(title, fontsize=9.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print("wrote", out)


def main():
    bars([("full model (T + R)", "lgb"), ("temporal only (no relational)", "T_only"),
          ("T + R + capture-rate features", "T_C_R"), ("T + capture-rate features", "T_C"),
          ("- own PRs / issues", "no_R_own"), ("- feedback received (reviews, comments)", "no_R_recv"),
          ("- repository neighbourhood", "no_R_nb"), ("- review demand", "no_R_demand"),
          ("- ownership / organisation", "no_R_owner"), ("- push details", "no_R_push"),
          ("RDL: GNN on the entity graph", "gnn_time"),
          ("hybrid LightGBM + GNN (in-sample weights)", "hybrid_lgb_gnn_time")],
         "Q2: relational information (LightGBM ablation, '-' = group removed)", "figures/ablation_q2.png")
    bars([("full model (T + R)", "lgb"), ("- window counts", "no_T_win"), ("- weekly lags", "no_T_lag"),
          ("- exponential decay", "no_T_ewm"), ("- recency (since last / first)", "no_T_rec"),
          ("- regularity (weeks active)", "no_T_reg"), ("- habits / trend", "no_T_hab"),
          ("only plain window counts", "basic_counts"), ("recency in calendar days (not recorded days)", "raw_age"),
          ("unrecorded weeks used as negatives (no mask)", "no_mask"), ("recency-weighted training (hl 8 w)", "hl8"),
          ("only 12 training weeks", "weeks12"), ("GNN with time encoding", "gnn_time"),
          ("GNN without time encoding", "gnn_notime")],
         "Q3: temporal information (ablation)", "figures/ablation_q3.png")

    # per-week, per-activity F1 of the full model vs baseline
    full = load("lgb")
    if full:
        rows = []
        for wk, r in full["validation"].items():
            for k, v in r["per_activity"].items():
                rows.append({"week": wk[:10], "activity": k, "model": v, "copy": r["copy_per"].get(k)})
        df = pd.DataFrame(rows)
        piv = df.pivot_table(index="activity", columns="week", values=["model", "copy"])
        piv.round(3).to_csv("figures/per_activity_f1.csv")
        print(piv.round(3).to_string())



def per_week():
    """Per validation week: copy-last-week baseline, LightGBM, GNN (with / without time), hybrid."""
    import numpy as np
    runs = [("copy last week", None), ("LightGBM (T + R)", "lgb"), ("GNN, no time enc.", "gnn_notime"),
            ("GNN + time enc.", "gnn_time"), ("hybrid (in-sample w)", "hybrid_lgb_gnn_time")]
    base = load("lgb")
    weeks = list(base["validation"])
    vals = {}
    for lab, t in runs:
        r = base if t is None else load(t)
        if r is None:
            continue
        vals[lab] = [r["validation"][w]["copy_last_week" if t is None else "model"] for w in weeks]
    fig, ax = plt.subplots(figsize=(7.6, 3.0), dpi=200)
    cols = ["#999999", "#2a78b5", "#e6b8a2", "#d1652f", "#3b9a5a"]
    n = len(vals); wd = 0.8 / n
    for j, (lab, v) in enumerate(vals.items()):
        x = np.arange(len(weeks)) + (j - (n - 1) / 2) * wd
        ax.bar(x, v, wd, label=lab, color=cols[j])
        for xi, vi in zip(x, v):
            ax.text(xi, vi + 0.004, f"{vi:.3f}", ha="center", fontsize=5.5, rotation=90)
    note = {0: "6 activities", 1: "5 (push unrecorded)", 2: "5 (push unrecorded)"}
    ax.set_xticks(range(len(weeks)), [f"cutoff {w[:10]} 00:00 UTC\n{note.get(i, '')}" for i, w in enumerate(weeks)],
                  fontsize=7)
    ax.set_ylabel("Score", fontsize=8); ax.set_ylim(0.25, max(max(v) for v in vals.values()) + 0.06)
    ax.legend(fontsize=6.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig("figures/per_week.png"); plt.close(fig)
    print("wrote figures/per_week.png")


if __name__ == "__main__":
    main()
    per_week()
