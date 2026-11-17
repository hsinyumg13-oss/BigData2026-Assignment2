"""Figure: weekly share of the normal GH Archive volume recorded, per event type."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

cap = pd.read_csv("data/coverage_capture.csv", index_col=0, parse_dates=True)
w = cap.resample("W").mean()
fig, ax = plt.subplots(figsize=(10, 3.6), dpi=200)
names = {"push": "push", "open_pr": "open_pr / merge", "open_issue": "open_issue", "issue_comment": "issue_comment",
         "pr_review": "pr_review", "pr_review_comment": "pr_review_comment"}
colors = ["#2a78b5", "#d1652f", "#3b9a5a", "#8a5cc2", "#c4407a", "#7a7a7a"]
for (k, n), col in zip(names.items(), colors):
    ax.plot(w.index, w[k], label=n, color=col, lw=2 if k == "push" else 1.4)
ax.axhline(1, color="#999", lw=0.8, ls="--")
ax.axvspan(pd.Timestamp("2025-12-01"), pd.Timestamp("2026-05-31"), color="#eee", zorder=0)
ax.text(pd.Timestamp("2026-02-15"), 1.95, "reference period (= 1.0)", ha="center", fontsize=8, color="#666")
ax.set_ylim(0, 2.1); ax.set_ylabel("recorded / normal volume")
ax.set_title("GH Archive capture rate per event type (cohort events, weekly mean)", fontsize=10)
ax.legend(ncol=6, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), frameon=False)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig("figures/capture_rate.png"); print("ok")
