"""Feature and label construction on the 8 relational tables.

A sample is (user, cutoff c). Features use only rows whose ROW TIME is before c, and the
"known only later" columns (merged_at, merged_by_id, closed_at, state_reason, dismissed_at)
only when their value is before c. Labels: did the user do activity k during [S, S + 7 days),
where S = c + gap (gap = time between the end of the data and the start of the target week).

Feature groups
  T  temporal  : multi-window counts, weekly lags, exp-decayed counts, recency, regularity
  R  relational: joins across tables (own PRs/issues -> others' reviews/comments, repo
                 neighbourhood activity, review demand, maintainer role, ownership)
"""
import numpy as np
import pandas as pd

TARGETS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
KINDS = TARGETS + ["merge_pr"]
DAY = pd.Timedelta(days=1)
WINDOWS = [1, 3, 7, 14, 28, 56, 112]
N_LAGS = 8
HALF_LIVES = [3, 14, 60]


def load_tables(db):
    names = ["Users", "Repositories", "PullRequests", "Issues", "IssueComments",
             "PRReviews", "PRReviewComments", "Pushes"]
    return {n: pd.read_parquet(f"{db}/{n}.parquet") for n in names}


ROW_TIME = {"Users": "first_seen_at", "Repositories": "first_seen_at", "PullRequests": "first_seen_at",
            "Issues": "first_seen_at", "IssueComments": "created_at", "PRReviews": "first_seen_at",
            "PRReviewComments": "created_at", "Pushes": "created_at"}
LATER = {"PullRequests": ["merged_at", "merged_by_id", "closed_at"], "Issues": ["closed_at", "state_reason"],
         "PRReviews": ["dismissed_at"]}
LATER_TIME = {"merged_at": "merged_at", "merged_by_id": "merged_at", "closed_at": "closed_at",
              "state_reason": "closed_at", "dismissed_at": "dismissed_at"}


def cut_tables(T, end):
    """Tables as they would be if the data ended at `end` (UTC): rows whose row time is >= end
    are dropped and the 'known only later' values after `end` are set to null. A re-run on data
    downloaded after the deadline therefore gives the same features."""
    out = {}
    for n, df in T.items():
        df = df[df[ROW_TIME[n]] < end].copy()
        masks = {col: df[LATER_TIME[col]].notna() & (df[LATER_TIME[col]] >= end) for col in LATER.get(n, [])}
        for col, m in masks.items():   # masks first: merged_by_id depends on merged_at
            df.loc[m, col] = None
        out[n] = df
    return out


def activity_log(T):
    """One long table of user activities: user, kind, time (event time), avail (row time),
    repo, obj (PR / issue id the activity is about, if any)."""
    pr, iss, ic = T["PullRequests"], T["Issues"], T["IssueComments"]
    rv, rc, ps = T["PRReviews"], T["PRReviewComments"], T["Pushes"]
    parts = []

    def add(df, user, kind, time, avail, obj=None):
        d = pd.DataFrame({"user": df[user], "kind": kind, "time": df[time],
                          "avail": df[avail], "repo": df["repo_id"],
                          "obj": df[obj] if obj else -1})
        parts.append(d.dropna(subset=["user", "time"]))

    add(pr[pr.created_at.notna()], "author_id", "open_pr", "created_at", "first_seen_at", "pr_id")
    issues_only = iss[~iss.is_pull_request.astype(bool)]
    add(issues_only, "author_id", "open_issue", "created_at", "first_seen_at", "issue_id")
    add(ic, "author_id", "issue_comment", "created_at", "created_at", "issue_id")
    add(rv, "reviewer_id", "pr_review", "submitted_at", "first_seen_at", "pr_id")
    add(rc, "author_id", "pr_review_comment", "created_at", "created_at", "pr_id")
    add(ps, "user_id", "push", "created_at", "created_at")
    # extra (non-target) activity: merging someone's PR; merged_at is "known only later",
    # so it is available only from merged_at on (avail = merged_at)
    m = pr[pr.merged_by_id.notna() & pr.merged_at.notna()]
    add(m, "merged_by_id", "merge_pr", "merged_at", "merged_at", "pr_id")
    log = pd.concat(parts, ignore_index=True)
    log["user"] = log["user"].astype(np.int64)
    log["avail"] = log[["avail", "time"]].max(axis=1)  # a row can't be used before it happened
    return log


COV = None  # daily coverage table (coverage.py); None = assume fully covered


CAP = None  # daily capture rate per kind (share of the normal GH Archive volume)


def set_coverage(cov, cap=None):
    global COV, CAP
    COV, CAP = cov, cap


def capture_features(c):
    """Snapshot-level: how much of the normal volume GH Archive recorded recently. The same
    for every user at a cutoff; it tells the model how many events it can expect to see."""
    f = {}
    if CAP is None:
        return f
    for w in [7, 28]:
        days = pd.date_range((c - w * DAY).normalize(), c.normalize() - DAY)
        m = CAP.reindex(days).mean()
        for k in KINDS:
            f[f"cap{w}_{k}"] = float(m.get(k, 1.0)) if pd.notna(m.get(k)) else 1.0
    return f


def _cum(times, kind):
    """Covered days from the start of the data up to each time (fraction of day included)."""
    if COV is None or kind not in COV:
        return (times - pd.Timestamp("2025-12-01")) / DAY
    c = COV[kind].values
    cum = pd.Series(c.cumsum() - c, index=COV.index)  # covered days before each day
    day = times.dt.normalize() if hasattr(times, "dt") else times.normalize()
    frac = (times - day) / DAY
    base = pd.Series(day).map(cum).fillna(cum.iloc[-1] + c[-1]).values
    rate = pd.Series(day).map(COV[kind]).fillna(1.0).values
    return base + frac * rate


RAW_AGE = False  # ablation switch: calendar-day age instead of recorded-day age


def covered_age(L, c):
    """Age of each row in COVERED days of its own kind: gaps in GH Archive are skipped, so
    a missing week of PushEvents does not make every user look inactive."""
    if RAW_AGE:
        return ((c - L.time) / DAY).values
    age = np.empty(len(L))
    for k, idx in L.groupby("kind").indices.items():
        tc = _cum(pd.Series([c]), k)[0]
        age[idx] = tc - _cum(L.time.iloc[idx].reset_index(drop=True), k)
    return age


def label_coverage(S):
    """Share of the target week that GH Archive recorded, per target."""
    if COV is None:
        return pd.Series(1.0, index=TARGETS)
    days = pd.date_range(S.normalize(), (S + 7 * DAY).normalize(), freq="D")
    w = pd.Series([max(0, min(1, ((min(d + DAY, S + 7 * DAY) - max(d, S)) / DAY))) for d in days], index=days)
    cv = COV.reindex(days).fillna(1.0)
    return pd.Series({k: float((cv[k] * w).sum() / w.sum()) for k in TARGETS})


def labels(log, users, S):
    E = S + 7 * DAY
    w = log[(log.time >= S) & (log.time < E) & log.kind.isin(TARGETS)]
    y = pd.crosstab(w.user, w.kind).reindex(index=users, columns=TARGETS, fill_value=0)
    return (y > 0).astype(np.int8)


def _counts(df, users, key, prefix):
    c = df.groupby(["user", key]).size().unstack(fill_value=0).reindex(columns=KINDS, fill_value=0)
    c.columns = [f"{prefix}{k}" for k in c.columns]
    return c.reindex(users, fill_value=0)


def temporal_features(L, users, c):
    """L = activity rows available before c."""
    L = L.reset_index(drop=True)
    L = L.assign(age=covered_age(L, c), rawage=((c - L.time) / DAY).values)
    out = []
    for w in WINDOWS:
        out.append(_counts(L[L.age < w], users, "kind", f"cnt{w}_"))
    for k in range(1, N_LAGS + 1):  # weekly lags (lag1 = last 7 days)
        out.append(_counts(L[(L.age >= 7 * (k - 1)) & (L.age < 7 * k)], users, "kind", f"lag{k}_"))
    for h in HALF_LIVES:  # exponentially decayed counts: recent rows count more
        d = L.assign(wt=np.power(0.5, L.age / h)).groupby(["user", "kind"]).wt.sum().unstack(fill_value=0).reindex(columns=KINDS, fill_value=0)
        d.columns = [f"ewm{h}_{k}" for k in d.columns]
        out.append(d.reindex(users, fill_value=0))
    last = L.groupby(["user", "kind"]).age.min().unstack().reindex(columns=KINDS)
    last.columns = [f"since_{k}" for k in last.columns]
    out.append(last.reindex(users).fillna(999))
    first = L.groupby(["user", "kind"]).age.max().unstack().reindex(columns=KINDS)
    first.columns = [f"first_{k}" for k in first.columns]
    out.append(first.reindex(users).fillna(-1))
    # regularity: in how many of the last 4 / 8 / 16 weeks was the activity done
    L16 = L[L.age < 112].assign(wk=(L.age // 7).astype(int))
    for nw in [4, 8, 16]:
        d = L16[L16.wk < nw].groupby(["user", "kind"]).wk.nunique().unstack(fill_value=0).reindex(columns=KINDS, fill_value=0) / nw
        d.columns = [f"wkfrac{nw}_{k}" for k in d.columns]
        out.append(d.reindex(users, fill_value=0))
    # whole-user activity
    tgt = L[L.kind.isin(TARGETS)]
    g = pd.DataFrame(index=users)
    for w in [7, 28, 112]:
        x = tgt[tgt.age < w]
        g[f"any_cnt{w}"] = x.groupby("user").size().reindex(users, fill_value=0)
        g[f"active_days{w}"] = x.assign(d=x.age.astype(int)).groupby("user").d.nunique().reindex(users, fill_value=0)
        g[f"n_repos{w}"] = x.groupby("user").repo.nunique().reindex(users, fill_value=0)
        g[f"n_kinds{w}"] = x.groupby("user").kind.nunique().reindex(users, fill_value=0)
    g["since_any"] = tgt.groupby("user").age.min().reindex(users).fillna(999)
    # hour-of-day / weekday habits over the last 112 days (Taiwan time)
    x = tgt[tgt.age < 112]
    tw = x.time + pd.Timedelta(hours=8)
    g["since_any_raw"] = tgt.groupby("user").rawage.min().reindex(users).fillna(999)
    g["weekend_share"] = (tw.dt.dayofweek >= 5).groupby(x.user).mean().reindex(users).fillna(0)
    g["night_share"] = ((tw.dt.hour < 7) | (tw.dt.hour >= 22)).groupby(x.user).mean().reindex(users).fillna(0)
    # trend: last 2 weeks vs the 6 weeks before
    a = tgt[tgt.age < 14].groupby("user").size().reindex(users, fill_value=0) / 2
    b = tgt[(tgt.age >= 14) & (tgt.age < 56)].groupby("user").size().reindex(users, fill_value=0) / 6
    g["trend_2v6"] = (a + 1) / (b + 1)
    out.append(g)
    X = pd.concat(out, axis=1)
    # capture-normalized counts: counts divided by the recorded share of the volume
    cf = capture_features(c)
    for k, v in cf.items():
        X[k] = v
    if cf:
        for k in KINDS:
            for w in [7, 28]:
                X[f"ncnt{w}_{k}"] = X[f"cnt{w}_{k}"] / max(cf[f"cap{w}_{k}"], 0.02)
    return X


def relational_features(T, L, users, c):
    """Features that follow foreign keys across tables, as known at cutoff c."""
    pr, iss, ic = T["PullRequests"], T["Issues"], T["IssueComments"]
    rv, rc, ps = T["PRReviews"], T["PRReviewComments"], T["Pushes"]
    rp, us = T["Repositories"], T["Users"]
    f = pd.DataFrame(index=users)
    uset = set(users)

    pr_c = pr[pr.first_seen_at < c]
    closed = pr_c.closed_at.notna() & (pr_c.closed_at < c)
    merged = pr_c.merged_at.notna() & (pr_c.merged_at < c)
    own = pr_c[pr_c.author_id.isin(uset)]
    recent_own = own[own.created_at >= c - 60 * DAY]
    f["own_pr_open"] = recent_own[~closed.reindex(recent_own.index)].groupby("author_id").size().reindex(users, fill_value=0)
    m28 = own[merged.reindex(own.index) & (own.merged_at >= c - 28 * DAY)]
    f["own_pr_merged28"] = m28.groupby("author_id").size().reindex(users, fill_value=0)
    f["own_pr_draft_share"] = recent_own.draft.astype(float).groupby(recent_own.author_id).mean().reindex(users).fillna(0)
    f["own_pr_cross_share"] = recent_own.is_cross_repo.astype(float).groupby(recent_own.author_id).mean().reindex(users).fillna(0)

    # PullRequests -> PRReviews / PRReviewComments by OTHER users on the user's PRs (last 7/28 d)
    pr_author = pr.set_index("pr_id").author_id
    for name, tab, who, tcol in [("rev", rv, "reviewer_id", "submitted_at"),
                                 ("revcmt", rc, "author_id", "created_at")]:
        x = tab[(tab[tcol] < c) & (tab[tcol] >= c - 28 * DAY)].copy()
        x["pr_author"] = x.pr_id.map(pr_author)
        x = x[x.pr_author.notna() & (x.pr_author != x[who])]
        for w in [7, 28]:
            y = x[x[tcol] >= c - w * DAY]
            f[f"recv_{name}{w}"] = y.groupby("pr_author").size().reindex(users, fill_value=0)
    x = rv[(rv.submitted_at < c) & (rv.submitted_at >= c - 28 * DAY)].copy()
    x["pr_author"] = x.pr_id.map(pr_author)
    x = x[x.pr_author.notna() & (x.pr_author != x.reviewer_id)]
    f["recv_changes_req28"] = x[x.state == "changes_requested"].groupby("pr_author").size().reindex(users, fill_value=0)

    # Issues -> IssueComments by others on the user's issues / PR conversations
    iss_author = iss.set_index("issue_id").author_id
    x = ic[(ic.created_at < c) & (ic.created_at >= c - 28 * DAY)].copy()
    x["iss_author"] = x.issue_id.map(iss_author)
    x = x[x.iss_author.notna() & (x.iss_author != x.author_id)]
    for w in [7, 28]:
        f[f"recv_icmt{w}"] = x[x.created_at >= c - w * DAY].groupby("iss_author").size().reindex(users, fill_value=0)
    iss_c = iss[(iss.first_seen_at < c) & ~iss.is_pull_request.astype(bool) & iss.author_id.isin(uset)]
    iss_closed = iss_c.closed_at.notna() & (iss_c.closed_at < c)
    f["own_issue_open"] = iss_c[~iss_closed & (iss_c.created_at >= c - 60 * DAY)].groupby("author_id").size().reindex(users, fill_value=0)

    # Repository neighbourhood: activity of OTHER users in the repos the user touched (28 d)
    L28 = L[(L.time >= c - 28 * DAY) & L.kind.isin(TARGETS)]
    ur = L28[["user", "repo"]].drop_duplicates()
    L7 = L28[L28.time >= c - 7 * DAY]
    repo_kind = L7.groupby(["repo", "kind"]).size().unstack(fill_value=0)
    repo_users = L7.groupby("repo").user.nunique()
    m = ur.join(repo_kind, on="repo").join(repo_users.rename("repo_nusers"), on="repo").fillna(0)
    own7 = L7.groupby(["user", "repo", "kind"]).size().unstack(fill_value=0)
    m = m.set_index(["user", "repo"])
    for k in TARGETS:
        if k not in m:
            m[k] = 0
        mine = own7[k].reindex(m.index, fill_value=0) if k in own7 else 0
        m[f"nb_{k}"] = m[k] - mine  # others' activity only
    nb = m.groupby(level="user")[[f"nb_{k}" for k in TARGETS] + ["repo_nusers"]].sum()
    nb.columns = [f"repo7_{k}" for k in nb.columns]
    f = f.join(nb.reindex(users).fillna(0))
    f["repo_nusers_max"] = m.groupby(level="user").repo_nusers.max().reindex(users).fillna(0)

    # Review demand: PRs opened by others in the last 7 d, in repos where the user
    # reviewed or merged in the last 56 d
    rev_repos = L[(L.time >= c - 56 * DAY) & L.kind.isin(["pr_review", "merge_pr", "pr_review_comment"])][["user", "repo"]].drop_duplicates()
    new_pr = L[(L.kind == "open_pr") & (L.time >= c - 7 * DAY)]
    np_repo = new_pr.groupby("repo").size()
    np_auth = new_pr.groupby(["repo", "user"]).size()
    rr = rev_repos.assign(n=rev_repos.repo.map(np_repo).fillna(0))
    rr["own"] = [np_auth.get((r, u), 0) for r, u in zip(rr.repo, rr.user)]
    f["review_demand7"] = (rr.n - rr.own).groupby(rr.user).sum().reindex(users, fill_value=0)
    f["n_review_repos56"] = rr.groupby("user").size().reindex(users, fill_value=0)
    # open PRs (not closed before c) of others in those repos
    open_pr = pr_c[~closed & (pr_c.created_at >= c - 30 * DAY)]
    op_repo = open_pr.groupby("repo_id").size()
    f["open_pr_in_review_repos"] = rr.repo.map(op_repo).fillna(0).groupby(rr.user).sum().reindex(users, fill_value=0)

    # Ownership: Users.login = Repositories.owner_login (user-owned repos), org repos
    login = us.set_index("user_id").login
    rep = rp.set_index("repo_id")
    x = L28.assign(owner=L28.repo.map(rep.owner_login), org=L28.repo.map(rep.org_id))
    x["is_own"] = x.owner.values == x.user.map(login).values
    f["own_repo_share28"] = x.groupby("user").is_own.mean().reindex(users).fillna(0)
    f["org_repo_share28"] = x.org.notna().groupby(x.user).mean().reindex(users).fillna(0)

    # Pushes detail
    p = ps[(ps.created_at < c) & (ps.created_at >= c - 28 * DAY)]
    f["push_new_ref28"] = p.groupby("user_id").is_new_ref.sum().reindex(users, fill_value=0)
    f["push_branches28"] = p.groupby("user_id").branch.nunique().reindex(users, fill_value=0)
    f["push_tag28"] = (p.ref_type == "tag").groupby(p.user_id).sum().reindex(users, fill_value=0)
    # pushes on branches that are the head of an open own PR -> likely to keep working
    # collaborators: distinct other users who reviewed / commented on the user's PRs & issues
    u0 = us.set_index("user_id")
    f["user_age_days"] = ((c - users.map(u0.first_seen_at)) / DAY).values
    return f


def build(T, log, users, c, S=None):
    L = log[log.avail < c]
    X = pd.concat([temporal_features(L, users, c), relational_features(T, L, users, c)], axis=1)
    X = X.astype(np.float32)
    y = labels(log, users, S) if S is not None else None
    return X, y
