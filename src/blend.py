"""Hybrid model: per activity, p = w * p_LightGBM + (1 - w) * p_GNN.

w and the decision threshold are chosen on the validation weeks only (pooled F1); the same
values are then applied to the back-test week or to the test week.

  python src/blend.py --lgb lgb --gnn gnn_time                       # validation report
  python src/blend.py --lgb lgb --gnn gnn_time --out output_113701031.csv   # + final output
  python src/blend.py --lgb bt_lgb --gnn bt_gnn --backtest           # score the back-test week
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import features as F  # noqa: E402
from train import best_threshold, score  # noqa: E402

W_GRID = np.round(np.linspace(0, 1, 11), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lgb", default="lgb"); ap.add_argument("--gnn", default="gnn_time")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--db", default="data/db"); ap.add_argument("--test_input", default="submission/test_input.csv")
    a = ap.parse_args()
    C = Path(a.cache)
    L, G = pd.read_pickle(C / f"valprob_{a.lgb}.pkl"), pd.read_pickle(C / f"valprob_{a.gnn}.pkl")
    assert [str(x) for x in L["val"]] == [str(x) for x in G["val"]], "different validation weeks"

    w_best, thr = {}, {}
    for k in F.TARGETS:
        ys, pl, pg = [], [], []
        for Y, P, PG in zip(L["Y"], L["P"], G["P"]):
            if k not in Y:
                continue
            ys.append(Y[k].values); pl.append(P[k].values); pg.append(PG[k].reindex(Y.index).fillna(0).values)
        if not ys:
            w_best[k], thr[k] = 1.0, L["thr"][k]; continue
        y, pl, pg = map(np.concatenate, (ys, pl, pg))
        res = [(best_threshold(y, w * pl + (1 - w) * pg), w) for w in W_GRID]
        (t, f), w = max(res, key=lambda r: r[0][1])
        w_best[k], thr[k] = float(w), t
    print("weights w (1 = LightGBM only):", w_best)

    rep = {}
    for i, (Y, P, PG) in enumerate(zip(L["Y"], L["P"], G["P"])):
        B = pd.DataFrame({k: ((w_best[k] * P[k] + (1 - w_best[k]) * PG[k].reindex(Y.index).fillna(0)) >= thr[k]).astype(int)
                          for k in Y.columns})
        s, per, _ = score(Y, B)
        rep[str(L["val"][i])] = {"model": round(s, 4), "per_activity": per}
        print(f"VAL {L['val'][i]}: hybrid {s:.4f} {per}")
    mean = float(np.mean([r["model"] for r in rep.values()]))
    print(f"MEAN VAL hybrid {mean:.4f}")
    out = {"weights": w_best, "thr": thr, "mean_model": mean, "validation": rep}
    # hybrid as probabilities, for src/significance.py
    pd.to_pickle({"Y": L["Y"], "val": L["val"], "thr": thr,
                  "P": [pd.DataFrame({k: w_best[k] * P[k] + (1 - w_best[k]) * PG[k].reindex(Y.index).fillna(0)
                                      for k in Y.columns}) for Y, P, PG in zip(L["Y"], L["P"], G["P"])]},
                 C / f"valprob_hybrid_{a.lgb}_{a.gnn}.pkl")

    # leave-one-week-out: weights / thresholds tuned on the other validation weeks only (honest estimate)
    lowo = {"lgb": [], "gnn": [], "hybrid": []}
    n = len(L["Y"])
    for h in range(n):
        tr = [i for i in range(n) if i != h]
        Yh = L["Y"][h]; Bs = {m: {} for m in lowo}
        for k in Yh.columns:
            idx = [i for i in tr if k in L["Y"][i]]
            if not idx:      # push is recorded in one validation week only -> in-sample threshold
                idx = [h]
            y = np.concatenate([L["Y"][i][k].values for i in idx])
            pl = np.concatenate([L["P"][i][k].values for i in idx])
            pg = np.concatenate([G["P"][i][k].reindex(L["Y"][i].index).fillna(0).values for i in idx])
            plh, pgh = L["P"][h][k].values, G["P"][h][k].reindex(Yh.index).fillna(0).values
            for m, grid in [("lgb", [1.0]), ("gnn", [0.0]), ("hybrid", W_GRID)]:
                (t, _), w = max(((best_threshold(y, w * pl + (1 - w) * pg), w) for w in grid), key=lambda r: r[0][1])
                Bs[m][k] = ((w * plh + (1 - w) * pgh) >= t).astype(int)
        for m in lowo:
            lowo[m].append(score(Yh, pd.DataFrame(Bs[m], index=Yh.index))[0])
    out["lowo"] = {m: {"weeks": [round(x, 4) for x in v], "mean": round(float(np.mean(v)), 4)} for m, v in lowo.items()}
    print("LEAVE-ONE-WEEK-OUT", out["lowo"])

    if a.backtest:
        bl, bg = pd.read_pickle(C / f"btprob_{a.lgb}.pkl"), pd.read_pickle(C / f"btprob_{a.gnn}.pkl")
        Y, P = bl["Y"], bl["P"]
        PG = bg["P"].reindex(P.index).fillna(0)
        res = {}
        for name, wt in [("lgb", {k: 1.0 for k in w_best}), ("gnn", {k: 0.0 for k in w_best}), ("hybrid", w_best)]:
            th = L["thr"] if name == "lgb" else (G["thr"] if name == "gnn" else thr)
            B = pd.DataFrame({k: ((wt[k] * P[k] + (1 - wt[k]) * PG[k]) >= th[k]).astype(int) for k in Y.columns})
            s, per, _ = score(Y, B)
            res[name] = {"model": round(s, 4), "per": per}
            print(f"BACKTEST {name}: {s:.4f} {per}")
        out["backtest"] = res
    if a.out:
        fl, fg = pd.read_pickle(C / f"finalprob_{a.lgb}.pkl"), pd.read_pickle(C / f"finalprob_{a.gnn}.pkl")
        U = pd.read_parquet(f"{a.db}/Users.parquet", columns=["user_id", "github_user_id"])
        uid_of = U.set_index("github_user_id").user_id
        test = pd.read_csv(a.test_input)
        uid = test.UserID.map(uid_of)
        res = pd.DataFrame({"UserID": test.UserID})
        for k in F.TARGETS:
            p = w_best[k] * uid.map(fl["P"][k]).fillna(0) + (1 - w_best[k]) * uid.map(fg["P"][k]).fillna(0)
            res[k] = (p >= thr[k]).astype(int).values
        res.to_csv(a.out, index=False)
        print("wrote", a.out, "positive rate", res[F.TARGETS].mean().round(3).to_dict())
    json.dump(out, open(C / f"exp_hybrid_{a.lgb}_{a.gnn}{'_bt' if a.backtest else ''}.json", "w"),
              indent=1, default=str)


if __name__ == "__main__":
    main()
