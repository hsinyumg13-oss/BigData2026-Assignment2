"""Paired bootstrap over users on the back-test week: hybrid / GNN vs LightGBM (thresholds and
weights as chosen on the validation weeks before the back-test). Output: figures/bt_bootstrap.csv"""
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from significance import week_score

L, G = pd.read_pickle("cache/btprob_bt_lgb.pkl"), pd.read_pickle("cache/btprob_bt_gnn.pkl")
GN = pd.read_pickle("cache/btprob_bt_gnn_nt.pkl")
H = json.load(open("cache/exp_hybrid_bt_lgb_bt_gnn_bt.json"))
tl = pd.read_pickle("cache/valprob_bt_lgb.pkl")["thr"]; tg = pd.read_pickle("cache/valprob_bt_gnn.pkl")["thr"]
tgn = pd.read_pickle("cache/valprob_bt_gnn_nt.pkl")["thr"]
Y = L["Y"]; cols = list(Y.columns); y = Y[cols].values.astype(bool)
pg = G["P"].reindex(Y.index).fillna(0); pgn = GN["P"].reindex(Y.index).fillna(0)
B = {"lgb": np.column_stack([L["P"][k] >= tl[k] for k in cols]),
     "gnn": np.column_stack([pg[k] >= tg[k] for k in cols]),
     "gnn_notime": np.column_stack([pgn[k] >= tgn[k] for k in cols]),
     "hybrid": np.column_stack([(H["weights"][k] * L["P"][k] + (1 - H["weights"][k]) * pg[k]) >= H["thr"][k] for k in cols])}
rng = np.random.default_rng(0); n = len(y); rows = []
idx = [rng.integers(0, n, n) for _ in range(1000)]
for m in ["gnn", "gnn_notime", "hybrid"]:
    d = np.array([week_score(y[i], B[m][i]) - week_score(y[i], B["lgb"][i]) for i in idx])
    pt = week_score(y, B[m]) - week_score(y, B["lgb"])
    lo, hi = np.quantile(d, [0.025, 0.975])
    rows.append({"model": m, "score": round(week_score(y, B[m]), 4), "diff_vs_lgb": round(pt, 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4)})
print("lgb", round(week_score(y, B["lgb"]), 4))
df = pd.DataFrame(rows); print(df.to_string()); df.to_csv("figures/bt_bootstrap.csv", index=False)
