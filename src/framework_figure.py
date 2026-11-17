"""Q4: framework figure of the full pipeline."""
import html

import graphviz


def box(g, name, title, lines, color, width=None):
    rows = "".join(f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="11">{html.escape(l)}</FONT></TD></TR>' for l in lines)
    g.node(name, f'<<TABLE BORDER="0" CELLBORDER="0" CELLPADDING="3" BGCOLOR="{color}">'
                 f'<TR><TD><B>{html.escape(title)}</B></TD></TR>{rows}</TABLE>>')


def main(out="figures/framework"):
    g = graphviz.Digraph("framework", format="png")
    g.attr(rankdir="LR", dpi="200", nodesep="0.3", ranksep="0.4", fontname="Helvetica")
    g.attr("node", shape="box", style="rounded", fontname="Helvetica", color="#555555")
    g.attr("edge", color="#555555")

    box(g, "raw", "1. GH Archive", ["hourly files, UTC", "2025-12-01 00:00 ..", "2026-11-21 12:00 (fixed)",
                                    "keep cohort events", "(same filter as", "preprocess.py)"], "#EAF2F8")
    box(g, "db", "2. 8 relational tables", ["course preprocess.py", "Users, Repositories,", "PullRequests, Issues,",
                                            "IssueComments, PRReviews,", "PRReviewComments, Pushes"], "#EAF2F8")
    box(g, "cov", "3. Coverage / capture", ["events per day and type", "vs Dec-May volume",
                                            "-> unrecorded weeks masked", "-> capture rate feature"], "#FEF5E7")
    box(g, "snap", "4. Weekly snapshots (user, t)", ["rows with row time < t only", "'known later' values only if < t",
                                                     "label week [t+gap, t+gap+7d)", "gap = 4 h (as at the deadline)",
                                                     "users: >= 3 activities in 91 d", "(rule of the test list)"], "#FEF5E7")
    with g.subgraph(name="cluster_a") as f:
        f.attr(label="5a. Feature branch (LightGBM)", style="rounded,dashed", color="#888888", fontsize="12")
        box(f, "ft", "Temporal (Q3)", ["windows 1-112 d, 8 weekly lags", "exp. decay 3/14/60 d",
                                       "recency in RECORDED days", "regularity, habits, trend"], "#E8F8F5")
        box(f, "fr", "Relational (Q2)", ["FK joins: others' reviews /", "comments on own PRs & issues,",
                                         "repo neighbourhood, review", "demand, ownership, pushes"], "#E8F8F5")
        box(f, "lgb", "LightGBM x 6", ["one model per activity", "~37 weekly snapshots"], "#F4ECF7")
    with g.subgraph(name="cluster_b") as f:
        f.attr(label="5b. Graph branch (RDL)", style="rounded,dashed", color="#888888", fontsize="12")
        box(f, "graph", "Relational entity graph at t", ["every row = node, FK = edge", "rows of the last 56 d",
                                                         "column features +", "time encoding of (t - row time)"], "#E8F8F5")
        box(f, "gnn", "Hetero GraphSAGE", ["2 layers, mean+sum aggr.", "user head -> 6 logits",
                                           "masked BCE loss"], "#F4ECF7")
    box(g, "blend", "6. Hybrid + thresholds", ["p = w p_LGB + (1-w) p_GNN", "w, threshold per activity",
                                               "chosen on 3 validation weeks", "(rolling time split)"], "#F4ECF7")
    box(g, "out", "7. Prediction", ["retrain on all weeks", "features at 11/21 20:00 (TW)",
                                    "test week 11/22 - 11/29", "output_113701031.csv", "(4,465 users x 6)"], "#FDEDEC")

    g.edge("raw", "db"); g.edge("raw", "cov"); g.edge("db", "snap"); g.edge("cov", "snap")
    g.edge("snap", "ft"); g.edge("snap", "fr"); g.edge("snap", "graph")
    g.edge("ft", "lgb"); g.edge("fr", "lgb"); g.edge("graph", "gnn")
    g.edge("lgb", "blend"); g.edge("gnn", "blend"); g.edge("blend", "out")
    g.render(out, cleanup=True)
    print("wrote", out + ".png")


if __name__ == "__main__":
    main()
