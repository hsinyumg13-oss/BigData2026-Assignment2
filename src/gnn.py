"""Relational Deep Learning baseline: a heterogeneous temporal GNN directly on the 8 tables.

Every row of every table is a node, every foreign key is an edge (both directions), as in
RelBench / RDL (Lecture 5). For a cutoff c the graph holds the rows whose row time is < c
inside a window of W days (plus the entity rows they point to); "known only later" values
are used only if they are < c. Each node gets its own column features plus, optionally, a
time encoding of how long before c the row happened. A 2-layer heterogeneous GraphSAGE
passes messages along the foreign keys and a head on the Users nodes predicts the 6
activities of [c + gap, c + gap + 7 d).

  python src/gnn.py --db data/db --coverage data/coverage.csv --data_end "2026-11-21 12:00" \
      --target_start "2026-11-22 00:00" --tag gnn_time
  python src/gnn.py ... --no_time --tag gnn_notime         # ablation: no time encoding
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv

sys.path.insert(0, str(Path(__file__).parent))
import features as F  # noqa: E402
from train import TW, MIN_COV, best_threshold, population, score  # noqa: E402

DAY = pd.Timedelta(days=1)
torch.set_num_threads(2)


# --------------------------------------------------------------------------------------
# graph at cutoff c
# --------------------------------------------------------------------------------------
def time_enc(age_days, on=True):
    """[log-age, decays 1/7/28 d, sin/cos of log-age at 3 scales] (10 dims); zeros if off."""
    a = np.clip(np.nan_to_num(np.asarray(age_days, dtype=np.float64), nan=365.0), 0, None)
    la = np.log1p(a)
    cols = [la / 6, np.exp(-a / 1), np.exp(-a / 7), np.exp(-a / 28)]
    for f in [1.0, 2.0, 4.0]:
        cols += [np.sin(f * la), np.cos(f * la)]
    x = np.stack(cols, 1).astype(np.float32)
    return x if on else np.zeros_like(x)


def _age(c, t):
    return ((c - t) / DAY).to_numpy(dtype=np.float64, na_value=np.nan)


def _b(s):
    return s.fillna(False).astype(np.float32).to_numpy()


def build_graph(T, c, window, use_time):
    """HeteroData of the rows known at c. Returns (data, user_ids in node order)."""
    lo = c - window * DAY
    us, rp, pr, iss = T["Users"], T["Repositories"], T["PullRequests"], T["Issues"]
    ic, rv, rc, ps = T["IssueComments"], T["PRReviews"], T["PRReviewComments"], T["Pushes"]
    # event rows inside the window (row time in [lo, c))
    ic = ic[(ic.created_at >= lo) & (ic.created_at < c)]
    rv = rv[(rv.first_seen_at >= lo) & (rv.first_seen_at < c)]
    rc = rc[(rc.created_at >= lo) & (rc.created_at < c)]
    ps = ps[(ps.created_at >= lo) & (ps.created_at < c)]
    # Pushes are ~70% of all rows; rows of the same (user, repository, day) are merged into one
    # node (number of pushes, new ref, tag) so the graph fits in memory
    ps = (ps.assign(day=ps.created_at.dt.floor("D"), tag=(ps.ref_type == "tag"))
            .groupby(["user_id", "repo_id", "day"], as_index=False)
            .agg(n=("push_id", "size"), is_new_ref=("is_new_ref", "max"), tag=("tag", "max"),
                 created_at=("created_at", "max")))
    ps["push_id"] = np.arange(len(ps))
    # entity rows: first seen in the window, or referenced by an event in the window
    pr_ids = set(pr.pr_id[(pr.first_seen_at >= lo) & (pr.first_seen_at < c)]) | set(rv.pr_id) | set(rc.pr_id)
    iss_ids = set(iss.issue_id[(iss.first_seen_at >= lo) & (iss.first_seen_at < c)]) | set(ic.issue_id)
    pr = pr[pr.pr_id.isin(pr_ids) & (pr.first_seen_at < c)]
    iss = iss[iss.issue_id.isin(iss_ids) & (iss.first_seen_at < c)]
    pr_ids = set(pr.pr_id) | set(iss.pull_request_id.dropna().astype(np.int64))
    pr = T["PullRequests"][T["PullRequests"].pr_id.isin(pr_ids) & (T["PullRequests"].first_seen_at < c)]
    rv = rv[rv.pr_id.isin(set(pr.pr_id))]
    rc = rc[rc.pr_id.isin(set(pr.pr_id))]
    ic = ic[ic.issue_id.isin(set(iss.issue_id))]
    merged = pr.merged_at.notna() & (pr.merged_at < c)
    user_ids = set(us.user_id[us.is_cohort]) | set(pr.author_id.dropna()) | set(pr.merged_by_id[merged].dropna()) \
        | set(iss.author_id.dropna()) | set(ic.author_id.dropna()) | set(rv.reviewer_id.dropna()) \
        | set(rc.author_id.dropna()) | set(ps.user_id.dropna())
    us = us[us.user_id.isin(user_ids) & (us.first_seen_at < c)]
    repo_ids = set(pr.repo_id) | set(iss.repo_id) | set(ic.repo_id) | set(ps.repo_id) | set(rv.repo_id) | set(rc.repo_id)
    rp = rp[rp.repo_id.isin(repo_ids) & (rp.first_seen_at < c)]

    tables = {"users": (us, "user_id"), "repos": (rp, "repo_id"), "prs": (pr, "pr_id"), "issues": (iss, "issue_id"),
              "icomments": (ic, "comment_id"), "reviews": (rv, "review_id"), "rcomments": (rc, "comment_id"),
              "pushes": (ps, "push_id")}
    idx = {k: pd.Series(np.arange(len(df)), index=df[key].values) for k, (df, key) in tables.items()}
    te = lambda t: time_enc(_age(c, t), use_time)  # noqa: E731

    d = HeteroData()
    cl = pr.closed_at.notna() & (pr.closed_at < c)
    icl = iss.closed_at.notna() & (iss.closed_at < c)
    xs = {
        "users": np.c_[_b(us.is_cohort), _b(us.is_bot), te(us.first_seen_at)],
        "repos": np.c_[rp.org_id.notna().astype(np.float32).to_numpy(), te(rp.first_seen_at)],
        "prs": np.c_[_b(pr.is_cross_repo), _b(pr.draft), merged.astype(np.float32).to_numpy(),
                     cl.astype(np.float32).to_numpy(), pr.created_at.notna().astype(np.float32).to_numpy(),
                     te(pr.created_at.fillna(pr.first_seen_at)), te(pr.first_seen_at)],
        "issues": np.c_[_b(iss.is_pull_request), np.log1p(iss.num_labels.fillna(0).to_numpy(dtype=np.float32)),
                        icl.astype(np.float32).to_numpy(),
                        (icl & (iss.state_reason == "completed")).astype(np.float32).to_numpy(),
                        te(iss.created_at.fillna(iss.first_seen_at))],
        "icomments": np.c_[_b(ic.is_pull_request), te(ic.created_at)],
        "reviews": np.c_[(rv.state == "approved").astype(np.float32).to_numpy(),
                         (rv.state == "changes_requested").astype(np.float32).to_numpy(),
                         (rv.state == "commented").astype(np.float32).to_numpy(),
                         (rv.dismissed_at.notna() & (rv.dismissed_at < c)).astype(np.float32).to_numpy(),
                         te(rv.submitted_at.fillna(rv.first_seen_at))],
        "rcomments": np.c_[(rc.subject_type == "line").astype(np.float32).to_numpy(), te(rc.created_at)],
        "pushes": np.c_[np.log1p(ps.n.to_numpy(dtype=np.float32)), _b(ps.is_new_ref), _b(ps.tag),
                        te(ps.created_at)],
    }
    for k, x in xs.items():
        d[k].x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

    def edge(src, df, fk, dst, rel, mask=None):
        if mask is not None:
            df = df[mask]
        s = idx[src].reindex(df[tables[src][1]].values).to_numpy()
        t = idx[dst].reindex(df[fk].values).to_numpy()
        ok = ~(np.isnan(s) | np.isnan(t))
        e = torch.from_numpy(np.stack([s[ok], t[ok]]).astype(np.int64))
        d[src, rel, dst].edge_index = e
        d[dst, "rev_" + rel, src].edge_index = e.flip(0)

    edge("prs", pr, "repo_id", "repos", "pr_repo")
    edge("prs", pr, "author_id", "users", "pr_author")
    edge("prs", pr, "merged_by_id", "users", "pr_merged_by", merged)
    edge("issues", iss, "repo_id", "repos", "issue_repo")
    edge("issues", iss, "author_id", "users", "issue_author")
    edge("issues", iss, "assignee_id", "users", "issue_assignee")
    edge("issues", iss, "pull_request_id", "prs", "issue_pr")
    edge("icomments", ic, "issue_id", "issues", "ic_issue")
    edge("icomments", ic, "author_id", "users", "ic_author")
    edge("reviews", rv, "pr_id", "prs", "rv_pr")
    edge("reviews", rv, "reviewer_id", "users", "rv_reviewer")
    edge("rcomments", rc, "pr_id", "prs", "rc_pr")
    edge("rcomments", rc, "author_id", "users", "rc_author")
    edge("rcomments", rc, "pull_request_review_id", "reviews", "rc_review")
    edge("pushes", ps, "user_id", "users", "push_user")
    edge("pushes", ps, "repo_id", "repos", "push_repo")
    return d, us.user_id.to_numpy()


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------
class RDL(nn.Module):
    def __init__(self, metadata, in_dims, hidden=64, layers=2, n_out=6):
        super().__init__()
        self.enc = nn.ModuleDict({t: nn.Sequential(nn.Linear(in_dims[t], hidden), nn.ReLU(),
                                                   nn.Linear(hidden, hidden)) for t in metadata[0]})
        self.convs = nn.ModuleList([HeteroConv({et: SAGEConv((-1, -1), hidden, aggr=["mean", "sum"])
                                                for et in metadata[1]}, aggr="sum") for _ in range(layers)])
        self.norms = nn.ModuleList([nn.ModuleDict({t: nn.LayerNorm(hidden) for t in metadata[0]})
                                    for _ in range(layers)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))

    def forward(self, d):
        h = {t: self.enc[t](d[t].x) for t in d.node_types}
        for conv, norm in zip(self.convs, self.norms):
            out = conv(h, d.edge_index_dict)
            h = {t: norm[t](torch.relu(out[t]) + h[t]) if t in out else h[t] for t in h}
        return self.head(h["users"]), h["users"]


# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db")
    ap.add_argument("--coverage", default="data/coverage.csv")
    ap.add_argument("--target_start", required=True, help="Taiwan time")
    ap.add_argument("--data_end", default=None, help="UTC, exclusive")
    ap.add_argument("--test_input", default="submission/test_input.csv")
    ap.add_argument("--n_weeks", type=int, default=40)
    ap.add_argument("--n_val", type=int, default=3)
    ap.add_argument("--window", type=int, default=56)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--no_time", action="store_true", help="ablation: no time encoding")
    ap.add_argument("--score_final", action="store_true", help="back-test: score the week after --data_end")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--tag", default="gnn")
    a = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)

    T = F.load_tables(a.db)
    if a.data_end and not a.score_final:
        T = F.cut_tables(T, pd.Timestamp(a.data_end))
    if Path(a.coverage).exists():
        F.set_coverage(pd.read_csv(a.coverage, index_col=0, parse_dates=True))
    log = F.activity_log(T)
    U = T["Users"]
    cohort = pd.Index(U[U.is_cohort].user_id.values)
    c_final = pd.Timestamp(a.data_end) if a.data_end else log.time.max().floor("h") + pd.Timedelta(hours=1)
    gap = pd.Timestamp(a.target_start) - TW - c_final
    snaps = sorted(c for c in (c_final - pd.Timedelta(days=7 * k) for k in range(1, a.n_weeks + 1))
                   if c + gap + 7 * DAY <= c_final and c - a.window * DAY >= pd.Timestamp("2025-12-01"))
    val, tr = snaps[-a.n_val:], snaps[:-a.n_val]
    print(f"data end {c_final}, gap {gap}, {len(tr)} train snapshots, val {val}", flush=True)

    gdir = Path(a.cache) / f"graphs_{c_final:%Y%m%d%H}_w{a.window}{'_nt' if a.no_time else ''}{'_bt' if a.score_final else ''}"
    gdir.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(a.test_input)
    test_uids = U[U.github_user_id.isin(test.UserID)].user_id.values

    def sample(c, final=False):
        p = gdir / f"{c:%Y%m%d%H}{'_final' if final else ''}.pt"
        if p.exists():
            return torch.load(p, weights_only=False)
        t0 = time.time()
        d, uids = build_graph(T, c, a.window, not a.no_time)
        seeds = test_uids if final else population(log, cohort, c, 91).values
        pos = pd.Series(np.arange(len(uids)), index=uids)
        seeds = np.array([s for s in seeds if s in pos.index])
        if final:
            y, cov = None, None
        else:
            y = torch.tensor(F.labels(log, pd.Index(seeds), c + gap).values, dtype=torch.float32)
            cov = torch.tensor([(F.label_coverage(c + gap)[k] >= MIN_COV) for k in F.TARGETS], dtype=torch.float32)
        s = {"d": d, "idx": torch.tensor(pos[seeds].values), "uid": seeds, "y": y, "cov": cov}
        torch.save(s, p)
        print(f"  graph {c}: {sum(d[t].num_nodes for t in d.node_types):,} nodes, "
              f"{sum(e.shape[1] for e in d.edge_index_dict.values()):,} edges, {len(seeds)} seeds "
              f"({time.time() - t0:.0f}s)", flush=True)
        return s

    def fit(train_snaps):
        first = sample(train_snaps[0])
        meta = (first["d"].node_types, first["d"].edge_types)
        model = RDL(meta, {t: first["d"][t].x.shape[1] for t in meta[0]}, a.hidden)
        with torch.no_grad():
            model(first["d"])  # lazy init
        opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
        for ep in range(a.epochs):
            model.train(); tot = 0.0
            for i in np.random.permutation(len(train_snaps)):
                s = sample(train_snaps[i])
                logit, _ = model(s["d"])
                l = nn.functional.binary_cross_entropy_with_logits(logit[s["idx"]], s["y"], reduction="none")
                l = (l * s["cov"]).sum() / (s["cov"].sum() * len(s["idx"]) + 1e-9)
                opt.zero_grad(); l.backward(); opt.step(); tot += float(l)
            print(f"  epoch {ep}: loss {tot / len(train_snaps):.4f}", flush=True)
        return model

    def predict(model, s):
        model.eval()
        with torch.no_grad():
            logit, emb = model(s["d"])
        return torch.sigmoid(logit[s["idx"]]).numpy(), emb[s["idx"]].numpy()

    for c in snaps:
        sample(c)
    sample(c_final, final=not a.score_final)
    del T, log
    import gc; gc.collect()

    # validation: train on the weeks before the validation weeks, tune thresholds on them
    model = fit(tr)
    Ys, Ps = [], []
    for vc in val:
        s = sample(vc)
        p, _ = predict(model, s)
        Ys.append(pd.DataFrame(s["y"].numpy(), columns=F.TARGETS, index=s["uid"]))
        Ps.append(pd.DataFrame(p, columns=F.TARGETS, index=s["uid"]))
    thr = {}
    for j, k in enumerate(F.TARGETS):
        ok = [i for i, vc in enumerate(val) if sample(vc)["cov"][j] > 0]
        thr[k] = best_threshold(np.concatenate([Ys[i][k] for i in ok]), np.concatenate([Ps[i][k] for i in ok]))[0] if ok else 0.5
    rep = {}
    for i, vc in enumerate(val):
        cols = [k for j, k in enumerate(F.TARGETS) if sample(vc)["cov"][j] > 0]
        B = pd.DataFrame({k: (Ps[i][k] >= thr[k]).astype(int) for k in cols})
        sc, per, _ = score(Ys[i][cols], B)
        rep[str(vc)] = {"model": round(sc, 4), "per_activity": per}
        print(f"VAL {vc}: {sc:.4f} {per}", flush=True)
    mean = float(np.mean([r["model"] for r in rep.values()]))
    print(f"MEAN VAL {a.tag} {mean:.4f}  thresholds {({k: round(v, 3) for k, v in thr.items()})}")
    out = {"tag": a.tag, "no_time": a.no_time, "window": a.window, "mean_model": mean, "validation": rep, "thr": thr}
    pd.to_pickle({"Y": Ys, "P": Ps, "val": val, "thr": thr}, Path(a.cache) / f"valprob_{a.tag}.pkl")

    # final: train on all snapshots; predict the final cutoff (test users or back-test week)
    model = fit(snaps)
    if a.score_final:
        s = sample(c_final, final=False)
        p, _ = predict(model, s)
        cols = [k for j, k in enumerate(F.TARGETS) if s["cov"][j] > 0]
        P = pd.DataFrame(p, columns=F.TARGETS)[cols]
        B = pd.DataFrame({k: (P[k] >= thr[k]).astype(int) for k in cols})
        sc, per, _ = score(pd.DataFrame(s["y"].numpy(), columns=F.TARGETS)[cols], B)
        print(f"BACKTEST {a.tag} week from {c_final + gap} UTC: {sc:.4f} {per}")
        out["backtest"] = {"model": sc, "per": per}
        pd.to_pickle({"uid": s["uid"], "P": pd.DataFrame(p, columns=F.TARGETS, index=s["uid"])},
                     Path(a.cache) / f"btprob_{a.tag}.pkl")
    else:
        s = sample(c_final, final=True)
        p, emb = predict(model, s)
        pd.to_pickle({"P": pd.DataFrame(p, columns=F.TARGETS, index=s["uid"]), "emb": emb},
                     Path(a.cache) / f"finalprob_{a.tag}.pkl")
    json.dump(out, open(Path(a.cache) / f"exp_{a.tag}.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
