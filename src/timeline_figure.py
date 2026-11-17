"""Q1: dataset time period - data range, training snapshots, validation weeks, gap, test week."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

TW = pd.Timedelta(hours=8)
START = pd.Timestamp("2025-12-01 00:00")          # UTC
END = pd.Timestamp("2026-11-21 12:00")            # UTC, exclusive (= 11/21 20:00 Taiwan)
TEST = pd.Timestamp("2026-11-22 00:00") - TW      # UTC
N_WEEKS, N_VAL = 40, 3
gap = TEST - END
D7 = pd.Timedelta(days=7)
snaps = sorted(c for c in (END - k * D7 for k in range(1, N_WEEKS + 1)) if c + gap + D7 <= END)
val, tr = snaps[-N_VAL:], snaps[:-N_VAL]

fig, ax = plt.subplots(figsize=(11, 3.4), dpi=200)
y = {"data": 4, "feat": 3, "train": 2, "val": 1, "test": 0}
ax.barh(y["data"], END - START, left=START, height=0.5, color="#2a78b5")
ax.text(START + (END - START) / 2, y["data"], "GH Archive hours used: 2025-12-01 00:00 - 2026-11-21 12:00 UTC  (8,532 files)",
        ha="center", va="center", color="white", fontsize=8.5, weight="bold")
ax.barh(y["feat"], tr[0] - START, left=START, height=0.5, color="#bcd3e8")
ax.text(START + (tr[0] - START) / 2, y["feat"], f"warm-up ({(tr[0] - START).days} d of history)", ha="center",
        va="center", fontsize=7.5)
ax.text(tr[0] + pd.Timedelta(days=3), y["feat"], "features at cutoff t: only rows with row time < t "
        "(windows up to 112 d, 'known later' values only if < t)", va="center", fontsize=7.5)
for c in tr:
    ax.barh(y["train"], D7, left=c + gap, height=0.5, color="#3b9a5a", edgecolor="white", lw=0.5)
for c in val:
    ax.barh(y["val"], D7, left=c + gap, height=0.5, color="#d1652f", edgecolor="white", lw=0.5)
ax.barh(y["test"], D7, left=TEST, height=0.5, color="#c4407a")
ax.text(tr[0] + gap, y["train"] + 0.42, f"{len(tr)} training label weeks (cutoff t every 7 days from {tr[0]:%Y-%m-%d})",
        fontsize=7.5, va="bottom")
ax.text(val[0] + gap - pd.Timedelta(days=2), y["val"], f"{len(val)} validation weeks", fontsize=7.5, va="center", ha="right")
ax.text(TEST - pd.Timedelta(days=2), y["test"], "test week 11/22 - 11/29 (Taiwan)", fontsize=7.5, va="center", ha="right")
ax.axvline(END, color="#444", ls="--", lw=0.8)
ax.text(END, 4.55, "data end\n11/21 20:00 TW", fontsize=7, ha="center", va="bottom")
ax.set_yticks(list(y.values()), ["data", "features", "train", "validation", "test"], fontsize=8)
ax.set_ylim(-0.6, 5.2)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
ax.tick_params(axis="x", labelsize=7.5)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.set_title("Dataset time period and time split (gap of 4 h between data end and label week, as at the deadline)",
             fontsize=9.5)
fig.tight_layout()
fig.savefig("figures/timeline.png")
print("train weeks", len(tr), tr[0], "..", tr[-1], "val", val, "gap", gap)
