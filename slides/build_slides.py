"""Fill the 16-slide skeleton (TA template, duplicated question pages) with the report content.
Numbers are read from cache/exp_*.json so the slides match the experiments.
Usage: python slides/build_slides.py skel.pptx out.pptx [github_url]
"""
import copy
import json
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
NAVY, ORANGE, GRAY, GREEN = RGBColor(0x34, 0x43, 0x94), RGBColor(0xD1, 0x65, 0x2F), RGBColor(0x55, 0x55, 0x55), RGBColor(0x3B, 0x9A, 0x5A)
FONT = "Helvetica Neue"


def J(tag):
    p = ROOT / "cache" / f"exp_{tag}.json"
    return json.load(open(p)) if p.exists() else None


def _set(para, t, size=None):
    rs = para.runs
    rs[0].text = t
    for r in rs[1:]:
        r.text = ""
    if size:
        rs[0].font.size = Pt(size)


def header(s, title, sub, sub_size=15):
    box = [sh for sh in s.shapes if sh.has_text_frame and "Question" in sh.text_frame.text][0]
    ps = box.text_frame.paragraphs
    if title is not None:
        _set(ps[0], title)
    _set(ps[1], sub if sub is not None else ps[1].text, sub_size)


def text(s, x, y, w, h, paras, size=11.5):
    """paras: list of str | (str, dict). dict keys: b, color, size, lvl (bullet level 0/1), bullet(bool), i"""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    for side in ("left", "right", "top", "bottom"):
        setattr(tf, f"margin_{side}", Inches(0.04))
    first = True
    for p in paras:
        t, o = (p, {}) if isinstance(p, str) else p
        para = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        runs = t if isinstance(t, list) else [(t, {})]
        for rt, ro in runs:
            oo = {**o, **ro}
            r = para.add_run(); r.text = rt
            f = r.font; f.name = FONT; f.size = Pt(oo.get("size", size)); f.bold = oo.get("b", False)
            f.italic = oo.get("i", False); f.color.rgb = oo.get("color", GRAY if not oo.get("b") else NAVY)
        para.space_after = Pt(o.get("after", 3))
        if o.get("bullet", False) or "lvl" in o:
            lvl = o.get("lvl", 0)
            pPr = para._p.get_or_add_pPr()
            pPr.set("marL", str(int(Inches(0.18 + 0.2 * lvl)))); pPr.set("indent", str(int(-Inches(0.15))))
            bu = pPr.makeelement(qn("a:buChar"), {"char": "•" if lvl == 0 else "–"})
            pPr.append(bu)
    return tb


def pic(s, name, x, y, w=None, h=None):
    kw = {}
    if w: kw["width"] = Inches(w)
    if h: kw["height"] = Inches(h)
    p = s.shapes.add_picture(str(FIG / name), Inches(x), Inches(y), **kw)
    return p


def table(s, data, x, y, w, colw, size=9.5, rowh=0.24, bold_rows=(0,), hl_rows=()):
    rows, cols = len(data), len(data[0])
    t = s.shapes.add_table(rows, cols, Inches(x), Inches(y), Inches(w), Inches(rowh * rows)).table
    for j, cw in enumerate(colw):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(data):
        t.rows[i].height = Inches(rowh)
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = str(v)
            c.margin_left = c.margin_right = Inches(0.05); c.margin_top = c.margin_bottom = Inches(0.01)
            c.fill.solid()
            c.fill.fore_color.rgb = NAVY if i == 0 else (RGBColor(0xFB, 0xE9, 0xDF) if i in hl_rows else
                                                         (RGBColor(0xF2, 0xF3, 0xF8) if i % 2 else RGBColor(0xFF, 0xFF, 0xFF)))
            for p in c.text_frame.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(size); r.font.name = FONT
                    r.font.bold = i in bold_rows or i in hl_rows
                    r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF) if i == 0 else RGBColor(0x33, 0x33, 0x33)
    return t


def callout(s, x, y, w, h, big, small, color=ORANGE):
    text(s, x, y, w, h, [(big, {"b": True, "size": 26, "color": color, "after": 0}), (small, {"size": 9.5})])


def notes(s, t):
    s.notes_slide.notes_text_frame.text = t


def main(skel, out, url):
    P = Presentation(skel)
    S = list(P.slides)
    lgb, gnn, gnt, hyb = J("lgb"), J("gnn_time"), J("gnn_notime"), J("hybrid_lgb_gnn_time")
    bt = J("hybrid_bt_lgb_bt_gnn_bt")
    btnt = J("bt_gnn_nt")
    c = lgb["mean_copy"]
    lw = hyb["lowo"]

    # ---------------- 1 cover
    s = S[0]
    for sh in s.shapes:   # widen the title box so the subtitle stays on one line above the rule
        if sh.has_text_frame and sh.text_frame.text.startswith("Assignment 2"):
            sh.left, sh.width = Inches(0.6), Inches(8.8)
    text(s, 1.5, 3.0, 7.0, 0.8, [(f"Student ID: 113701031", {"b": True, "size": 14, "after": 2}),
                                 ([("GitHub: ", {"b": True}), (url, {"color": ORANGE})], {"size": 12})])
    for p in s.shapes[-1].text_frame.paragraphs:
        p.alignment = 2

    # ---------------- 2 overview
    s = S[1]
    header(s, "Overview", "Predict 6 activities of 4,465 GitHub users in the week 2026/11/22 - 11/29 (Taiwan)", 15)
    text(s, 0.45, 1.05, 5.0, 4.1, [
        ("Data", {"b": True, "size": 12.5}),
        ("GH Archive -> course preprocess.py -> 8 relational tables; 8,532 hourly files, 2025-12-01 .. 2026-11-21 20:00 (TW)", {"bullet": True}),
        ("GH Archive itself recorded only 10-30 % of the events since June (push ~0 % since late Oct.) - this shapes every design choice", {"bullet": True}),
        ("Method", {"b": True, "size": 12.5}),
        ("Feature branch: temporal + FK-join relational features per (user, cutoff t), LightGBM x 6", {"bullet": True}),
        ("Graph branch (RDL): every row a node, every FK an edge, heterogeneous GraphSAGE with a time encoding", {"bullet": True}),
        ("Weekly snapshots, rolling time split, per-activity F1 thresholds; hybrid of both branches", {"bullet": True}),
        ("Evaluation", {"b": True, "size": 12.5}),
        ("3 validation weeks + leave-one-week-out + paired bootstrap; back-test on the README week 9/27", {"bullet": True}),
    ], size=10.5)
    callout(s, 5.75, 1.15, 3.9, 0.8, f"{lgb['mean_model']:.3f} / {gnn['mean_model']:.3f}",
            f"mean validation Score, LightGBM / GNN  (copy last week {c:.3f})")
    if bt:
        b = bt["backtest"]
        callout(s, 5.75, 2.15, 3.9, 0.8, f"{max(b['lgb']['model'], b['gnn']['model'], b['hybrid']['model']):.3f}",
                "back-test week 9/27 (TA reference: GNN + time 0.548, copy last week 0.496)", GREEN)
    text(s, 5.75, 3.2, 3.9, 1.9, [
        ("Main findings", {"b": True, "size": 12}),
        ("Time encoding is what makes the GNN work (+0.031)", {"bullet": True}),
        ("Capture-rate covariates hurt under the GH Archive collapse (-0.026 when added to T + R)", {"bullet": True}),
        ("Most single feature groups are redundant (within noise)", {"bullet": True}),
        ("~2/3 of positives are 'new' activity - the hard part", {"bullet": True}),
    ], size=10)

    # ---------------- 3 Q1 schema
    s = S[2]
    header(s, None, None)
    pic(s, "schema_graph.png", 0.35, 1.05, h=4.25)
    text(s, 5.05, 1.15, 4.65, 4.1, [
        ("8 tables built by scripts/preprocess.py", {"b": True, "size": 12}),
        ("Nodes = tables; PK marked, each FK field points to the PK it references (20 FK edges incl. the self-loop PRReviewComments.in_reply_to_id)", {"bullet": True}),
        ("Row time (when a row can be used): first_seen_at for entity tables, created_at for comment / push rows", {"bullet": True}),
        ("'L' = known only later: merged_at, merged_by_id, closed_at, state_reason, dismissed_at -> used only if < t", {"bullet": True}),
        ("Rows in our data (cohort events)", {"b": True, "size": 11, "after": 2}),
    ], size=10)
    table(s, [["Table", "Rows", "Table", "Rows"],
              ["Users", "46,197", "IssueComments", "108,021"],
              ["Repositories", "111,025", "PRReviews", "92,702"],
              ["PullRequests", "309,787", "PRReviewComments", "67,193"],
              ["Issues", "177,265", "Pushes", "1,672,328"]], 5.1, 3.95, 4.5, [1.15, 0.9, 1.5, 0.95], size=9, rowh=0.22)

    # ---------------- 4 Q1 time period
    s = S[3]
    header(s, "(1%) Question 1 · Time period", "Fixed data window, weekly snapshots, time-based split with the deadline's 4-hour gap")
    pic(s, "timeline.png", 0.3, 1.0, w=9.4)
    table(s, [["", "UTC", "Taiwan time"],
              ["GH Archive hours used (8,532 files)", "2025-12-01 00:00 - 2026-11-21 12:00 (excl.)", "12-01 08:00 - 11-21 20:00"],
              ["Training label weeks (36 cutoffs)", "cutoffs 2026-02-14 12:00 ... 10-17 12:00", "+4 h gap, 7-day label week"],
              ["Validation label weeks (3), final run", "cutoffs 10-24, 10-31, 11-07 (12:00)", "rolling, after all training weeks"],
              ["Validation weeks of the report results", "cutoffs 10-19, 10-26, 11-02 (00:00), data end 11-16 00:00", "slides 6-15; same 4 h gap, 5.5 d earlier"],
              ["Test week", "2026-11-21 16:00 - 11-28 16:00", "11-22 00:00 - 11-29 00:00"]],
          0.45, 3.95, 9.1, [3.0, 3.4, 2.7], size=8.5, rowh=0.22)
    notes(s, "The end of the data is fixed so the TA's re-run downloads the same hours and reproduces the output.")

    # ---------------- 5 Q2 approach
    s = S[4]
    header(s, None, None)
    text(s, 0.45, 1.05, 4.5, 0.3, [("A. FK-join features (feature branch)", {"b": True, "size": 12.5})])
    table(s, [["Group", "Join path (all rows with row time < t)"],
              ["own PRs / issues", "PullRequests / Issues.author_id = user: open (not closed < t), merged in 28 d, draft / cross-repo share"],
              ["feedback received", "PRReviews, PRReviewComments -> PullRequests.pr_id -> author; IssueComments -> Issues -> author (others only, 7 / 28 d)"],
              ["repo neighbourhood", "repos the user acted in (28 d) -> OTHER users' activity of each type there in the last 7 d, # users"],
              ["review demand", "repos where user reviewed / merged (56 d) -> new PRs by others (7 d), open PRs"],
              ["ownership / org", "Users.login = Repositories.owner_login; Repositories.org_id"],
              ["push details", "Pushes: new refs, branches, tags (28 d)"],
              ["merge_pr", "PullRequests.merged_by_id, usable from merged_at on"]],
          0.45, 1.4, 4.6, [1.2, 3.4], size=8, rowh=0.42)
    text(s, 5.3, 1.05, 4.4, 0.3, [("B. Relational Deep Learning (graph branch)", {"b": True, "size": 12.5})])
    text(s, 5.3, 1.4, 4.4, 3.8, [
        ("Instead of hand-picked joins, the model learns them:", {}),
        ("every row of the 8 tables (row time in [t-56 d, t)) is a node, typed by its table", {"bullet": True}),
        ("every FK is an edge, plus the reverse edge (16 relation types x 2)", {"bullet": True}),
        ("node input = the row's columns (numbers, flags, categories) + time encoding", {"bullet": True}),
        ("pushes merged per (user, repo, day) to keep the graph in memory", {"bullet": True}),
        ("Hetero GraphSAGE, 2 layers, hidden 64, mean + sum aggregation, residual + LayerNorm", {"bullet": True}),
        ("User nodes -> 6 logits; BCE loss masked on unrecorded label weeks", {"bullet": True}),
        ("2 layers reach e.g. user -> PR -> reviews of others, user -> repo -> other users' PRs", {"i": True, "size": 9.5}),
    ], size=10)

    # ---------------- 6 Q2 results
    s = S[5]
    header(s, "(3%) Question 2 · Results", "LightGBM ablation (95% paired-bootstrap intervals) and the graph model")
    pic(s, "ablation_q2.png", 0.3, 1.0, h=4.3)
    text(s, 6.25, 1.1, 3.45, 4.1, [
        ("Findings", {"b": True, "size": 12}),
        (f"Hand-made FK features: no group is significant; T only = {J('T_only')['mean_model']:.3f} vs T + R = {lgb['mean_model']:.3f}. Relational counts are largely redundant with the user's own temporal history.", {"bullet": True}),
        (f"Learned joins (RDL GNN) reach the same level without any hand-made feature: {gnn['mean_model']:.3f}", {"bullet": True}),
        (f"Leave-one-week-out (weights & thresholds tuned on the other weeks): LightGBM {lw['lgb']['mean']:.3f}, GNN {lw['gnn']['mean']:.3f}, hybrid {lw['hybrid']['mean']:.3f}", {"bullet": True}),
        ("The in-sample hybrid gain (+0.012) shrinks to 0 in leave-one-week-out, but on the unseen back-test week 9/27 the hybrid beats LightGBM by +0.008 (95% CI 0.003 - 0.013)", {"bullet": True, "color": ORANGE}),
    ], size=9.5)

    # ---------------- 7 Q2 graph details / per-activity
    s = S[6]
    header(s, "(3%) Question 2 · Per activity", "No branch dominates; they make different kinds of errors")
    rows = [["Activity", "copy last wk", "LightGBM", "GNN + time", "GNN no time"]]
    acts = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]
    import numpy as np
    def per(r, k, key="per_activity"):
        v = [w[key][k] for w in r["validation"].values() if k in w[key]]
        return np.mean(v) if v else float("nan")
    for k in acts:
        rows.append([k, f"{per(lgb, k, 'copy_per'):.3f}", f"{per(lgb, k):.3f}", f"{per(gnn, k):.3f}", f"{per(gnt, k):.3f}"])
    table(s, rows, 0.45, 1.1, 5.4, [1.6, 0.95, 0.95, 0.95, 0.95], size=9.5, rowh=0.3)
    text(s, 0.45, 3.35, 5.4, 1.8, [
        ("Mean F1 over the validation weeks where the activity is recorded (push: week 10-19 only).", {"i": True, "size": 9}),
        ("Hybrid weight w (1 = LightGBM only) chosen per activity: " +
         ", ".join(f"{k} {v:.1f}" for k, v in hyb["weights"].items()), {"size": 9.5}),
    ], size=9.5)
    text(s, 6.1, 1.1, 3.6, 4.0, [
        ("Reading", {"b": True, "size": 12}),
        ("The GNN sees the actual rows around the user (which PRs got reviews, who else is active in the repo), the feature model only aggregate counts", {"bullet": True}),
        ("Without time encoding the GNN only knows that a row exists in the 56-day window, not how recent it is -> large loss on every activity", {"bullet": True}),
        ("Back-test 9/27: the GNN predicts far more positives (issue_comment 1,090 vs 445 true; LightGBM 183) -> higher macro-F1 (0.481 vs 0.449) but lower micro-F1 (0.575 vs 0.614). The hybrid sits in between", {"bullet": True}),
    ], size=9.5)

    # ---------------- 8 Q3 capture collapse
    s = S[7]
    header(s, None, None)
    pic(s, "capture_rate.png", 0.3, 1.15, w=6.4)
    text(s, 6.85, 1.15, 2.85, 4.1, [
        ("The time axis is broken", {"b": True, "size": 12}),
        ("Cohort events per day / Dec-May volume, per event type", {"size": 9.5, "i": True}),
        ("Since June GH Archive records only 10-30 % of the events (brief recovery in early Sept.)", {"bullet": True}),
        ("PushEvents ~0-3 % since 10/27", {"bullet": True}),
        ("-> a quiet week may be a recording gap, not an inactive user", {"bullet": True, "color": ORANGE}),
        ("-> labels in such weeks are false negatives", {"bullet": True, "color": ORANGE}),
        ("Same issue in the test week: the official labels come from the same archive", {"bullet": True}),
    ], size=9.5)
    text(s, 0.45, 3.65, 6.2, 1.5, [
        ("Daily coverage table (src/coverage.py) is used for", {"b": True, "size": 10.5}),
        ("masking: a (week, activity) label is used only if >= 70 % of that week was recorded", {"bullet": True}),
        ("recency in recorded days: unrecorded days do not count as 'days since last activity'", {"bullet": True}),
        ("population rule: >= 3 activities in 91 days reproduces test_input.csv (F1 0.996)", {"bullet": True}),
    ], size=9.5)

    # ---------------- 9 Q3 temporal design
    s = S[8]
    header(s, "(5%) Question 3 · Design", "Every row is placed on the time axis; features at t only see rows with row time < t")
    cards = [
        ("1. Point-in-time snapshots", "one sample per (user, weekly cutoff t); rows filtered by row time < t; 'known later' columns nulled if >= t (cut_tables); merges usable from merged_at"),
        ("2. Multi-scale windows", "counts per activity in 1, 3, 7, 14, 28, 56, 112 days + 8 weekly lags"),
        ("3. Decay & recency", "exponential decay, half-lives 3 / 14 / 60 d; days since last / first activity (in recorded days)"),
        ("4. Regularity & habits", "share of the last 4 / 8 / 16 weeks active; weekday / night share (Taiwan time); 2-week vs 6-week trend"),
        ("5. Labels & split", "label week [t + 4 h, t + 4 h + 7 d), as at the deadline; rolling time split; masked unrecorded weeks"),
        ("6. Time in the graph", "GNN node feature: log age, decays (1 / 7 / 28 d), sin / cos of t - row time at 3 scales"),
    ]
    why = ["Why: no leakage - exactly what is known at the deadline",
           "Why: short windows catch bursts, long windows the base rate",
           "Why: smooth recency without window edges; archive gaps do not look like inactivity",
           "Why: separates regular contributors from one-off bursts",
           "Why: validation mirrors the test situation; no training on unrecorded (false) negatives",
           "Why: without it a row from yesterday and one from 8 weeks ago look the same (-0.030 val, -0.044 back-test)"]
    for i, (h, b) in enumerate(cards):
        x = 0.45 + (i % 3) * 3.08; y = 1.1 + (i // 3) * 2.05
        sh = s.shapes.add_shape(1, Inches(x), Inches(y), Inches(2.9), Inches(1.9))
        sh.fill.solid(); sh.fill.fore_color.rgb = RGBColor(0xF2, 0xF3, 0xF8); sh.line.fill.background()
        sh.shadow.inherit = False
        text(s, x + 0.08, y + 0.08, 2.75, 1.75, [(h, {"b": True, "size": 12.5, "after": 6}), (b, {"size": 10.5, "after": 6}),
                                                 (why[i], {"size": 9.5, "i": True, "color": ORANGE})])

    # ---------------- 10 Q3 ablation
    s = S[9]
    header(s, "(5%) Question 3 · Ablation", "Ablation (LightGBM unless noted), 95% paired-bootstrap intervals")
    pic(s, "ablation_q3.png", 0.3, 1.0, h=4.3)
    text(s, 5.6, 1.1, 4.1, 4.1, [
        ("Significant", {"b": True, "size": 12}),
        (f"GNN time encoding: {gnt['mean_model']:.3f} -> {gnn['mean_model']:.3f} (+{gnn['mean_model'] - gnt['mean_model']:.3f})", {"bullet": True}),
        ("Recency-weighted training (half-life 8 weeks) hurts: older weeks still help", {"bullet": True}),
        ("Capture-rate covariates (same for every user in a week): T + R + C is 0.026 below T + R here, 0.043 below on the validation weeks before 9/27, yet 0.014 above on the 9/27 back-test -> unstable, left out (chosen on validation)", {"bullet": True}),
        ("Within noise", {"b": True, "size": 12}),
        ("Each temporal sub-group alone: the windows, lags and decays carry overlapping information; only plain window counts is as good", {"bullet": True}),
        ("Recorded-day recency and label masking: about 0 on these weeks (still kept: they avoid a known bias)", {"bullet": True}),
        ("Only 12 training weeks: -0.009 (borderline)", {"bullet": True}),
    ], size=9.5)

    # ---------------- 11 Q3 per week
    s = S[10]
    header(s, "(5%) Question 3 · Stability", "Score per validation week; the level moves with GH Archive's capture rate")
    pic(s, "per_week.png", 0.3, 1.05, w=6.5)
    rows = [["", "wk 1", "wk 2", "wk 3", "mean"]]
    for name in ["lgb", "gnn", "hybrid"]:
        rows.append([{"lgb": "LightGBM", "gnn": "GNN + time", "hybrid": "hybrid"}[name]] +
                    [f"{v:.3f}" for v in lw[name]["weeks"]] + [f"{lw[name]['mean']:.3f}"])
    text(s, 0.45, 3.75, 6.3, 0.3, [("Leave-one-week-out (thresholds and weights tuned on the other 2 weeks)", {"b": True, "size": 10})])
    table(s, rows, 0.45, 4.05, 6.3, [1.7, 1.15, 1.15, 1.15, 1.15], size=9, rowh=0.25)
    text(s, 7.0, 1.1, 2.7, 4.1, [
        ("Observations", {"b": True, "size": 12}),
        ("Week 1 is easiest: push is still recorded there and push is the most predictable activity", {"bullet": True}),
        ("LightGBM and GNN + time beat copy-last-week in every week (+0.05 to +0.10)", {"bullet": True}),
        ("Ranking of the models changes from week to week -> differences < 0.01 should not be over-read", {"bullet": True}),
        ("Push (week 1 only) keeps an in-sample threshold in leave-one-week-out", {"bullet": True, "size": 8.5, "i": True}),
    ], size=9.5)

    # ---------------- 12 Q4 framework
    s = S[11]
    header(s, None, None)
    pic(s, "framework.png", 0.25, 1.25, w=9.5)
    text(s, 0.45, 4.15, 9.1, 1.1, [
        ("Steps 1-3 are data preparation (GH Archive -> course preprocess.py -> 8 tables, coverage); step 4 builds point-in-time snapshots; "
         "steps 5a / 5b are the two ways of modelling the tables (Q2) with the temporal design of Q3; step 6 picks thresholds (and mixing weights) "
         "on the validation weeks only; step 7 retrains on all weeks and predicts from the data up to 11/21 20:00 (TW).", {"size": 10}),
    ])

    # ---------------- 13 Q4 pipeline + reproducibility
    s = S[12]
    header(s, "(5%) Question 4 · Pipeline", "./run_all.sh  ->  output_113701031.csv  (every step can be run on its own)")
    table(s, [["Step", "Script", "What it does"],
              ["1", "src/download_filter.py", "download each GH Archive hour, keep the lines preprocess.extract_hour keeps (cohort, 6 types, no bots)"],
              ["2", "scripts/preprocess.py", "course script, unchanged -> 8 parquet tables"],
              ["3", "src/coverage.py", "events per day and type -> coverage / capture tables"],
              ["4", "src/train.py", "snapshots, features, rolling validation, thresholds, LightGBM x 6, final probabilities"],
              ["5", "src/gnn.py", "entity graph per snapshot, hetero GraphSAGE, same split, final probabilities"],
              ["6", "src/blend.py", "per-activity weight / threshold from validation, write output_113701031.csv"],
              ["7", "scripts/evaluate.py", "official format check"]],
          0.45, 1.05, 9.1, [0.5, 2.0, 6.6], size=8.5, rowh=0.27)
    text(s, 0.45, 3.35, 9.1, 1.9, [
        ("Reproducibility", {"b": True, "size": 12}),
        ("DATA_END fixed at 2026-11-21 12:00 UTC: the same 8,532 hour files are downloaded at any later time, and train / gnn drop every row (and every 'known later' value) after it", {"bullet": True}),
        ("LightGBM deterministic mode (seed 0, 2 threads, force_row_wise); GNN fixed seed; exact package versions in requirements.txt", {"bullet": True}),
        ("All report experiments: ./run_experiments.sh (resumable), figures from src/*_figure.py", {"bullet": True}),
    ], size=9.5)

    # ---------------- 14 Q4 results
    s = S[13]
    header(s, "(5%) Question 4 · Results", "3 validation weeks and a back-test on the README week 2026/09/27 - 10/04")
    rows = [["Model", "Validation (in-sample thr.)", "Leave-one-week-out", "Back-test 9/27"]]
    b = bt["backtest"] if bt else None
    rows.append(["copy last week", f"{c:.4f}", "-", "0.4994  (TA: 0.4961)"])
    rows.append(["TA: GNN without time", "-", "-", "0.3570 (TA)"])
    rows.append(["TA: GNN with time", "-", "-", "0.5482 (TA)"])
    rows.append(["ours: LightGBM (T + R)", f"{lgb['mean_model']:.4f}", f"{lw['lgb']['mean']:.4f}", f"{b['lgb']['model']:.4f}" if b else "running"])
    import pandas as pd
    bb = pd.read_csv(ROOT / "figures" / "bt_bootstrap.csv").set_index("model")
    rows.append(["ours: GNN without time", f"{gnt['mean_model']:.4f}", "-", f"{bb.loc['gnn_notime', 'score']:.4f}"])
    rows.append(["ours: GNN + time (RDL)", f"{gnn['mean_model']:.4f}", f"{lw['gnn']['mean']:.4f}", f"{b['gnn']['model']:.4f}" if b else "running"])
    rows.append(["ours: hybrid", f"{hyb['mean_model']:.4f}", f"{lw['hybrid']['mean']:.4f}", f"{b['hybrid']['model']:.4f}" if b else "running"])
    table(s, rows, 0.45, 1.1, 9.1, [2.6, 2.2, 2.1, 2.2], size=9.5, rowh=0.3, hl_rows=(FINAL_ROW,))
    text(s, 0.45, 3.65, 9.1, 1.6, FINAL_TEXT(lgb, gnn, hyb, lw, b), size=9.5)

    # ---------------- 15 Q4 error analysis
    s = S[14]
    header(s, "(5%) Question 4 · Error analysis", "Where the model fails: users who start an activity they did not do the week before")
    pic(s, "error_analysis.png", 0.3, 1.05, w=9.4)
    text(s, 0.45, 4.25, 9.1, 1.0, [
        ("58-76 % of the positives are users who did not do that activity in the previous week; recall on them is only 17-30 %, against 70-90 % for repeat users. "
         "Part of this is the archive: with 10-30 % capture, 'did not do it last week' is often 'not recorded last week'. "
         "Push is predicted best (F1 0.68): it is the most habitual activity. pr_review_comment is the hardest (rare, bursty).", {"size": 10}),
    ])

    # ---------------- 16 conclusion
    s = S[15]
    header(s, "Conclusion and limitations", "What worked, what did not, and what we would do next")
    text(s, 0.45, 1.1, 4.5, 4.1, [
        ("What we learned", {"b": True, "size": 12}),
        ("Temporal information is the core signal; in a graph model it must be given explicitly (time encoding +0.031)", {"bullet": True}),
        ("Relational structure is useful when the model can learn the joins (RDL GNN = feature model without hand-made features); hand-made FK aggregates add little over a strong temporal history", {"bullet": True}),
        ("With a broken data source, honest evaluation matters more than one more feature: leave-one-week-out shrank the in-sample hybrid gain (+0.012 -> 0); the truly unseen back-test week still shows +0.008", {"bullet": True}),
    ], size=10)
    text(s, 5.2, 1.1, 4.5, 4.1, [
        ("Limitations / next steps", {"b": True, "size": 12}),
        ("Only 3 validation weeks, and push is recorded in one of them -> wide uncertainty", {"bullet": True}),
        ("The test labels come from the same under-recording archive: the test score depends on GH Archive's capture that week, which no model can see", {"bullet": True}),
        ("GNN: temporal sampling of neighbours, longer windows, several seeds averaged", {"bullet": True}),
        ("Calibrate thresholds to the expected positive rate instead of the validation F1 optimum", {"bullet": True}),
    ], size=10)
    P.save(out)
    print("saved", out)


# set after the back-test: which row of the results table is the submitted model, and its text
FINAL_ROW = 7


def FINAL_TEXT(lgb, gnn, hyb, lw, b):
    import pandas as pd
    bb = pd.read_csv(ROOT / "figures" / "bt_bootstrap.csv").set_index("model")
    h = bb.loc["hybrid"]; g = bb.loc["gnn_notime"]
    return [
        ("Submitted model: hybrid (highlighted)", {"b": True, "size": 12}),
        (f"Back-test 9/27 (weights and thresholds from the 3 weeks before it, i.e. truly out of sample): hybrid {h.score:.4f} vs LightGBM {b['lgb']['model']:.4f}, "
         f"difference +{h.diff_vs_lgb:.3f} (95% paired-bootstrap CI {h.ci_low:.3f} - {h.ci_high:.3f}); on the Oct.-Nov. weeks hybrid = LightGBM (leave-one-week-out)", {"bullet": True}),
        (f"Time encoding again decisive: GNN without time {g.score:.4f} (below copy-last-week), with time {b['gnn']['model']:.4f}", {"bullet": True}),
        ("Our models stay ~0.01 below the TA's GNN + time reference (0.548) on 9/27, but beat copy-last-week by +0.04; our copy baseline (0.499) is close to the TA's (0.496), so the user populations are comparable but not identical", {"bullet": True}),
    ]


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "https://github.com/<repo link added before submission>")
