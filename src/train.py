"""Train one LightGBM per activity on weekly snapshots and predict the test week.

  python src/train.py --db data/db --coverage data/coverage.csv --data_end "2026-11-21 12:00" \
        --target_start "2026-11-22 00:00" --out output_113701031.csv

Time handling (no leakage):
  * c_final = end of the data (last hour in the tables); target week starts at S_final
    (given in Taiwan time); gap = S_final - c_final.
  * training snapshots: c_k = c_final - 7k days, features at c_k, labels in
    [c_k + gap, c_k + gap + 7 d); a snapshot is used only if its whole label week is in the data.
  * validation = the most recent snapshots (time split, never random).
  * a (snapshot, activity) whose label week GH Archive did not record (coverage < MIN_COV)
    is dropped for that activity: its zeros are missing data, not real negatives.
"""
import argparse
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent))
import features as F  # noqa: E402

TW = pd.Timedelta(hours=8)
MIN_COV = 0.7
PARAMS = dict(objective="binary", learning_rate=0.03, num_leaves=31, min_child_samples=50,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=0, num_threads=2, deterministic=True, force_row_wise=True)
N_ROUNDS = 600


def population(log, users, c, active_days, min_events=3):
    """Users eligible at cutoff c, chosen like the official test list: cohort users with at
    least `min_events` target activities in the last `active_days` days (this rule reproduces
    test_input.csv with F1 = 0.996: >= 3 activities since 2026-08-05)."""
    t = log[(log.time < c) & (log.time >= c - pd.Timedelta(days=active_days)) & log.kind.isin(F.TARGETS)]
    n = t.groupby("user").size()
    return users[users.isin(n[n >= min_events].index)]


SUBGROUPS = {  # finer feature groups for the Q2 / Q3 ablations
    "T_win": ("cnt",), "T_lag": ("lag",), "T_ewm": ("ewm",), "T_rec": ("since_", "first_"),
    "T_reg": ("wkfrac",), "T_hab": ("any_cnt", "active_days", "n_repos", "n_kinds", "weekend_share",
                                    "night_share", "trend_"),
    "C": ("cap7_", "cap28_", "ncnt"),
    "R_own": ("own_pr", "own_issue"), "R_recv": ("recv_",), "R_nb": ("repo7_", "repo_nusers"),
    "R_demand": ("review_demand", "n_review_repos", "open_pr_in"), "R_owner": ("own_repo", "org_repo"),
    "R_push": ("push_",), "R_age": ("user_age",),
}


def subgroup(col):
    best = ""
    for g, pre in SUBGROUPS.items():   # longest matching prefix wins (ncnt vs cnt, own_repo vs own_pr)
        for p in pre:
            if col.startswith(p) and len(p) > len(best):
                best, name = p, g
    return name if best else "other"


def best_threshold(y, p):
    best = (0.5, 0.0)
    for th in np.unique(np.quantile(p, np.linspace(0.3, 0.995, 140))):
        f = f1_score(y, p >= th, zero_division=0)
        if f > best[1]:
            best = (float(th), f)
    return best


def score(Y, P):
    """(macro-F1 + micro-F1) / 2 over the activities present in Y (same as evaluate.py)."""
    cols = [k for k in F.TARGETS if k in Y]
    f1s = [f1_score(Y[k], P[k], zero_division=0) for k in cols]
    micro = f1_score(Y[cols].values.ravel(), P[cols].values.ravel(), zero_division=0)
    return (np.mean(f1s) + micro) / 2, dict(zip(cols, np.round(f1s, 4))), micro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db")
    ap.add_argument("--coverage", default="data/coverage.csv")
    ap.add_argument("--target_start", required=True, help="Taiwan time, e.g. '2026-11-12 00:00'")
    ap.add_argument("--test_input", default="submission/test_input.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_weeks", type=int, default=24)
    ap.add_argument("--n_val", type=int, default=3)
    ap.add_argument("--active_days", type=int, default=91)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--groups", default="T,R", help="feature groups: T temporal, C capture, R relational")
    ap.add_argument("--basic", action="store_true", help="T = plain window counts only (ablation)")
    ap.add_argument("--halflife", type=float, default=0, help="recency weight half-life in weeks (0 = off)")
    ap.add_argument("--no_final", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--drop", default="", help="comma list of feature subgroups to leave out (ablation), "
                                               "see SUBGROUPS")
    ap.add_argument("--no_mask", action="store_true",
                    help="ablation: train also on weeks GH Archive did not record (their zeros as negatives)")
    ap.add_argument("--raw_age", action="store_true",
                    help="ablation: measure recency in calendar days instead of recorded days")
    ap.add_argument("--data_end", default=None,
                    help="UTC hour where the data ends (exclusive). Fixed in run_all.sh so that a re-run "
                         "on data downloaded later gives the same output; also used for back-tests")
    ap.add_argument("--score_final", action="store_true",
                    help="backtest: score the final week (needs --data_end) on test-like users")
    a = ap.parse_args()

    F.RAW_AGE = a.raw_age
    T = F.load_tables(a.db)
    if a.data_end and not a.score_final:   # ignore anything after the fixed end of the data
        T = F.cut_tables(T, pd.Timestamp(a.data_end))
    if Path(a.coverage).exists():
        cap = a.coverage.replace(".csv", "_capture.csv")
        F.set_coverage(pd.read_csv(a.coverage, index_col=0, parse_dates=True),
                       pd.read_csv(cap, index_col=0, parse_dates=True) if Path(cap).exists() else None)
    log = F.activity_log(T)
    U = T["Users"]
    cohort = pd.Index(U[U.is_cohort].user_id.values)
    gid = U.set_index("user_id").github_user_id

    c_final = log.time.max().floor("h") + pd.Timedelta(hours=1)
    if a.data_end:
        c_final = pd.Timestamp(a.data_end)
    S_final = pd.Timestamp(a.target_start) - TW
    gap = S_final - c_final
    print(f"data end {c_final} UTC, target week starts {S_final} UTC, gap {gap}")

    snaps = []
    for k in range(1, a.n_weeks + 1):
        c = c_final - pd.Timedelta(days=7 * k)
        if c + gap + pd.Timedelta(days=7) <= c_final:
            snaps.append(c)
    snaps = sorted(snaps)
    val_snaps, tr_snaps = snaps[-a.n_val:], snaps[:-a.n_val]
    print(f"{len(tr_snaps)} train snapshots {tr_snaps[0]} .. {tr_snaps[-1]}; validation {val_snaps}")

    test = pd.read_csv(a.test_input)
    test_uids = pd.Index(U[U.github_user_id.isin(test.UserID)].user_id.values)
    cache = Path(a.cache) / (f"feats_{c_final:%Y%m%d%H}_gap{int(gap / pd.Timedelta(hours=1))}_{a.n_weeks}"
                             f"{'_raw' if a.raw_age else ''}{'_bt' if a.score_final else ''}.pkl")
    if cache.exists():
        data = pd.read_pickle(cache)
        print("loaded", cache)
    else:
        data = None
    for c in ([] if data else snaps + [c_final]):
        if not data:
            data = {}
        S = c + gap
        final_unlabeled = c == c_final and not a.score_final
        users = test_uids if final_unlabeled else population(log, cohort, c, a.active_days)
        X, y = F.build(T, log, users, c, None if final_unlabeled else S)
        X["snapshot"] = len(data)
        data[c] = (X, y, None if final_unlabeled else F.label_coverage(S))
        cov = "" if c == c_final else " cov " + json.dumps(data[c][2].round(2).to_dict())
        print(f"  snapshot {c}: {len(users)} users" + ("" if y is None else f", pos rate {y.mean().round(3).to_dict()}") + cov)

    if not cache.exists():
        Path(a.cache).mkdir(exist_ok=True)
        pd.to_pickle(data, cache)
    feats = [col for col in data[c_final][0].columns if col != "snapshot"]
    groups = set(a.groups.split(","))

    def group(col):
        if col.startswith(("cap7_", "cap28_", "ncnt")):
            return "C"
        if col.startswith(("own_", "recv_", "repo7_", "repo_nusers", "review_demand", "n_review_repos",
                           "open_pr_in", "org_repo", "push_", "user_age")):
            return "R"
        return "T"
    feats = [f for f in feats if group(f) in groups]
    drop = set(filter(None, a.drop.split(",")))
    feats = [f for f in feats if subgroup(f) not in drop]
    if a.basic:
        feats = [f for f in feats if group(f) != "T" or f.startswith("cnt")]
    print(f"{len(feats)} features, groups {sorted(groups)}{' (basic T)' if a.basic else ''}, halflife {a.halflife}")

    def stack(cs, k):
        Xs, ys, ws = [], [], []
        for c in cs:
            X, y, cv = data[c]
            if cv[k] < MIN_COV and not a.no_mask:
                continue
            Xs.append(X[feats]); ys.append(y[k])
            age_w = (max(cs) - c) / pd.Timedelta(days=7)
            ws.append(np.full(len(X), 0.5 ** (age_w / a.halflife) if a.halflife else 1.0))
        return (pd.concat(Xs), pd.concat(ys), np.concatenate(ws)) if Xs else (None, None, None)

    # rolling validation: for each validation week, train on all earlier weeks
    thr, rep = {}, {}
    vals_Y, vals_P = [], []
    for i, vc in enumerate(val_snaps):
        prev = [c for c in snaps if c < vc]
        Xv, yv, cv = data[vc]
        P = pd.DataFrame(index=Xv.index)
        Yd = pd.DataFrame(index=Xv.index)
        for k in F.TARGETS:
            Xt, yt, wt = stack(prev, k)
            if Xt is None or cv[k] < MIN_COV:
                continue
            m = lgb.train(PARAMS, lgb.Dataset(Xt, yt, weight=wt), N_ROUNDS)
            P[k] = m.predict(Xv[feats]); Yd[k] = yv[k]
        vals_Y.append(Yd); vals_P.append(P)
    for k in F.TARGETS:
        ys = [Y[k] for Y in vals_Y if k in Y]; ps = [P[k] for P in vals_P if k in P]
        if not ys:
            # no recorded label in the validation weeks (e.g. push missing in GH Archive):
            # tune the threshold on the most recent earlier weeks that do have this activity
            ok = [c for c in snaps if data[c][2][k] >= MIN_COV and c < min(val_snaps)][-a.n_val:]
            for vc in ok:
                Xt, yt, wt = stack([c for c in snaps if c < vc], k)
                if Xt is None:
                    continue
                m = lgb.train(PARAMS, lgb.Dataset(Xt, yt, weight=wt), N_ROUNDS)
                ps.append(pd.Series(m.predict(data[vc][0][feats]))); ys.append(data[vc][1][k])
            if not ys:
                thr[k] = 0.5; continue
            print(f"  {k}: threshold tuned on earlier recorded weeks {[str(c) for c in ok]}")
        thr[k], _ = best_threshold(pd.concat(ys).values, pd.concat(ps).values)
    for i, (Y, P) in enumerate(zip(vals_Y, vals_P)):
        B = pd.DataFrame({k: (P[k] >= thr[k]).astype(int) for k in Y.columns})
        Xv = data[val_snaps[i]][0]
        copy = pd.DataFrame({k: (Xv[f"lag1_{k}"] > 0).astype(int) for k in Y.columns})
        s, per, mi = score(Y, B); s0, per0, mi0 = score(Y, copy)
        rep[str(val_snaps[i])] = {"model": round(s, 4), "per_activity": per, "copy_last_week": round(s0, 4), "copy_per": per0}
        print(f"VAL {val_snaps[i]}: model {s:.4f} {per} | copy-last-week {s0:.4f}")
    print("thresholds", {k: round(v, 3) for k, v in thr.items()})
    pd.to_pickle({"Y": vals_Y, "P": vals_P, "val": val_snaps, "thr": thr},
                 Path(a.cache) / f"valprob_{a.tag or 'lgb'}.pkl")
    mean_model = np.mean([r["model"] for r in rep.values()]); mean_copy = np.mean([r["copy_last_week"] for r in rep.values()])
    print(f"MEAN VAL model {mean_model:.4f} copy {mean_copy:.4f}")
    Path(a.cache).mkdir(exist_ok=True)
    json.dump({"tag": a.tag, "groups": a.groups, "basic": a.basic, "halflife": a.halflife, "n_feats": len(feats),
               "mean_model": mean_model, "mean_copy": mean_copy, "validation": rep},
              open(f"{a.cache}/exp_{a.tag or 'run'}.json", "w"), indent=1, default=str)
    if a.score_final:
        Xf, yf, cvf = data[c_final]
        P = pd.DataFrame(index=Xf.index); Yd = pd.DataFrame(index=Xf.index)
        for k in F.TARGETS:
            if cvf[k] < MIN_COV:
                continue
            Xt, yt, wt = stack(snaps, k)
            m = lgb.train(PARAMS, lgb.Dataset(Xt, yt, weight=wt), N_ROUNDS)
            P[k] = m.predict(Xf[feats]); Yd[k] = yf[k]
        pd.to_pickle({"P": P.copy(), "Y": Yd.copy()}, Path(a.cache) / f"btprob_{a.tag or 'lgb'}.pkl")
        P = pd.DataFrame({k: (P[k] >= thr[k]).astype(int) for k in P.columns})
        copy = pd.DataFrame({k: (Xf[f"lag1_{k}"] > 0).astype(int) for k in Yd.columns})
        s1, per1, mi1 = score(Yd, P); s0, per0, mi0 = score(Yd, copy)
        print(f"BACKTEST week from {c_final + gap} UTC, {len(Xf)} users: model {s1:.4f} {per1} | copy {s0:.4f} {per0}")
        json.dump({"model": s1, "per": per1, "copy": s0, "copy_per": per0, "users": len(Xf)},
                  open(f"{a.cache}/backtest_{a.tag}.json", "w"), indent=1, default=str)
        return
    if a.no_final:
        return

    # final: train on every snapshot, predict the test week
    Xf = data[c_final][0]
    uid_of = pd.Series(gid.index, index=gid.values)
    out = pd.DataFrame({"UserID": test.UserID})
    imps, probs = {}, {}
    for k in F.TARGETS:
        Xt, yt, wt = stack(snaps, k)
        m = lgb.train(PARAMS, lgb.Dataset(Xt, yt, weight=wt), N_ROUNDS)
        p = pd.Series(m.predict(Xf[feats]), index=Xf.index)
        probs[k] = p
        uid = test.UserID.map(uid_of)
        out[k] = (uid.map(p).fillna(0) >= thr[k]).astype(int)
        imps[k] = pd.Series(m.feature_importance("gain"), index=feats).sort_values(ascending=False).head(15).round(0).to_dict()
    out.to_csv(a.out, index=False)
    pd.to_pickle({"P": pd.DataFrame(probs), "thr": thr}, Path(a.cache) / f"finalprob_{a.tag or 'lgb'}.pkl")
    print("predicted positive rate", out[F.TARGETS].mean().round(3).to_dict())
    Path(a.cache).mkdir(exist_ok=True)
    json.dump({"validation": rep, "thresholds": thr, "importance": imps, "gap_hours": gap / pd.Timedelta(hours=1),
               "data_end": str(c_final)}, open(f"{a.cache}/report.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
