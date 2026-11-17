"""Q1: schema graph of the 8 tables (nodes) and their foreign keys (edges FK -> PK)."""
import html
import sys

import graphviz

TABLES = {
    "Users": [("user_id", "PK"), ("github_user_id", ""), ("login", ""), ("is_bot", ""),
              ("is_cohort", ""), ("first_seen_at", "T")],
    "Repositories": [("repo_id", "PK"), ("github_repo_id", ""), ("name", ""), ("owner_login", ""),
                     ("org_id", ""), ("org_login", ""), ("first_seen_at", "T")],
    "PullRequests": [("pr_id", "PK"), ("repo_id", "FK"), ("author_id", "FK"), ("merged_by_id", "FK*"),
                     ("number", ""), ("draft", ""), ("is_cross_repo", ""), ("created_at", ""),
                     ("merged_at", "L"), ("closed_at", "L"), ("first_seen_at", "T")],
    "Issues": [("issue_id", "PK"), ("repo_id", "FK"), ("author_id", "FK"), ("assignee_id", "FK"),
               ("pull_request_id", "FK"), ("is_pull_request", ""), ("num_labels", ""), ("created_at", ""),
               ("closed_at", "L"), ("state_reason", "L"), ("first_seen_at", "T")],
    "IssueComments": [("comment_id", "PK"), ("issue_id", "FK"), ("repo_id", "FK"), ("author_id", "FK"),
                      ("is_pull_request", ""), ("body", ""), ("created_at", "T")],
    "PRReviews": [("review_id", "PK"), ("pr_id", "FK"), ("repo_id", "FK"), ("reviewer_id", "FK"),
                  ("state", ""), ("submitted_at", ""), ("dismissed_at", "L"), ("first_seen_at", "T")],
    "PRReviewComments": [("comment_id", "PK"), ("pr_id", "FK"), ("repo_id", "FK"), ("author_id", "FK"),
                         ("pull_request_review_id", "FK"), ("in_reply_to_id", "FK"), ("path", ""),
                         ("subject_type", ""), ("created_at", "T")],
    "Pushes": [("push_id", "PK"), ("user_id", "FK"), ("repo_id", "FK"), ("ref_type", ""), ("branch", ""),
               ("is_new_ref", ""), ("created_at", "T")],
}

# (table, fk column) -> referenced table (its PK)
FKS = [
    ("PullRequests", "repo_id", "Repositories"), ("PullRequests", "author_id", "Users"),
    ("PullRequests", "merged_by_id", "Users"),
    ("Issues", "repo_id", "Repositories"), ("Issues", "author_id", "Users"),
    ("Issues", "assignee_id", "Users"), ("Issues", "pull_request_id", "PullRequests"),
    ("IssueComments", "issue_id", "Issues"), ("IssueComments", "repo_id", "Repositories"),
    ("IssueComments", "author_id", "Users"),
    ("PRReviews", "pr_id", "PullRequests"), ("PRReviews", "repo_id", "Repositories"),
    ("PRReviews", "reviewer_id", "Users"),
    ("PRReviewComments", "pr_id", "PullRequests"), ("PRReviewComments", "repo_id", "Repositories"),
    ("PRReviewComments", "author_id", "Users"), ("PRReviewComments", "pull_request_review_id", "PRReviews"),
    ("PRReviewComments", "in_reply_to_id", "PRReviewComments"),
    ("Pushes", "user_id", "Users"), ("Pushes", "repo_id", "Repositories"),
]
PK = {t: cols[0][0] for t, cols in TABLES.items()}
COLOR = {"Users": "#DCEBF7", "Repositories": "#DCEBF7", "PullRequests": "#E3F2E1", "Issues": "#E3F2E1",
         "IssueComments": "#FDF0DC", "PRReviews": "#FDF0DC", "PRReviewComments": "#FDF0DC", "Pushes": "#FDF0DC"}


def label(t):
    rows = [f'<TR><TD COLSPAN="2" BGCOLOR="{COLOR[t]}"><B>{t}</B></TD></TR>']
    for c, k in TABLES[t]:
        tag = {"PK": "<B>PK</B>", "FK": "FK", "FK*": "FK, L", "T": "row time", "L": "L"}.get(k, " ")
        name = f"<B><U>{c}</U></B>" if k == "PK" else (f"<I>{c}</I>" if k.startswith("FK") else html.escape(c))
        p1, p2 = (f' PORT="{c}"', "") if k == "PK" else ("", f' PORT="{c}"')
        rows.append(f'<TR><TD ALIGN="LEFT"{p1}>{name}</TD>'
                    f'<TD ALIGN="LEFT"{p2}><FONT POINT-SIZE="9" COLOR="#666666">{tag}</FONT></TD></TR>')
    return '<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="3">' + "".join(rows) + "</TABLE>>"


def main(out="figures/schema_graph"):
    g = graphviz.Digraph("schema", format="png")
    g.attr(rankdir="LR", dpi="200", nodesep="0.35", ranksep="1.4", splines="spline", fontname="Helvetica",
           label="PK = primary key (row number), FK = foreign key (edge FK -> referenced PK), "
                 "L = known only later (use only if < prediction time t)", labelloc="b", fontsize="11")
    g.attr("node", shape="plaintext", fontname="Helvetica", fontsize="10")
    g.attr("edge", color="#555555", arrowsize="0.7")
    # layers left -> right: child tables ... parent tables, so every FK edge points right
    layers = [["PRReviewComments", "IssueComments", "Pushes"], ["PRReviews", "Issues"],
              ["PullRequests"], ["Users", "Repositories"]]
    for i, layer in enumerate(layers):
        with g.subgraph(name=f"L{i}") as s:
            s.attr(rank="same")
            for t in layer:
                s.node(t, label(t))
    for t, c, r in FKS:
        if t == r:   # self reference (reply thread)
            g.edge(f"{t}:{c}:w", f"{r}:{PK[r]}:w", constraint="false")
        else:
            g.edge(f"{t}:{c}:e", f"{r}:{PK[r]}:w")
    g.render(out, cleanup=True)
    print("wrote", out + ".png", f"({len(TABLES)} tables, {len(FKS)} foreign keys)")


if __name__ == "__main__":
    main(*sys.argv[1:])
