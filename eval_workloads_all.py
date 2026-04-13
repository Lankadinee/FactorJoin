"""Evaluate FactorJoin on workloads_all.sql for both IMDB and STATS.

Produces:
1. perror_input files for E2E
2. q-error report on all queries
"""
import run_experiment  # noqa — circular import fix
import pickle
import re
import time as _time
import numpy as np
import os
import sys

PRICE_DIR = "/home/student.unimelb.edu.au/lrathuwadu/PRICE"
FJ_DIR = "/home/student.unimelb.edu.au/lrathuwadu/FactorJoin"
EST_DIR = "/home/student.unimelb.edu.au/lrathuwadu/estimates"
SUBPLAN_RE = re.compile(r"/\*\s*\(.*?\)\s*\*/")

# ---------- IMDB helpers ----------
def convert_imdb_query(sql):
    """Strip /* */ comments, normalize AND, strip parens from WHERE."""
    sql = SUBPLAN_RE.sub("", sql).strip()
    sql = re.sub(r"\s+and\s+", " AND ", sql, flags=re.IGNORECASE)
    m = re.search(r"\bWHERE\b", sql, re.IGNORECASE)
    if m:
        prefix = sql[:m.end()]
        body = sql[m.end():]
        body = body.replace("(", "").replace(")", "")
        sql = prefix + body
    return sql.rstrip(";").strip() + ";"

# ---------- STATS helpers ----------
STATS_ALIAS_MAP = {
    "st_b": "b", "st_c": "c", "st_p": "p", "st_ph": "ph",
    "st_pl": "pl", "st_t": "t", "st_u": "u", "st_v": "v",
}
STATS_COL_MAP = {
    "id": "Id", "userid": "UserId", "postid": "PostId",
    "creationdate": "CreationDate", "date": "Date",
    "score": "Score", "votetypeid": "VoteTypeId",
    "bountyamount": "BountyAmount", "posthistorytypeid": "PostHistoryTypeId",
    "posttypeid": "PostTypeId", "viewcount": "ViewCount",
    "owneruserid": "OwnerUserId", "answercount": "AnswerCount",
    "commentcount": "CommentCount", "favoritecount": "FavoriteCount",
    "reputation": "Reputation", "views": "Views",
    "upvotes": "UpVotes", "downvotes": "DownVotes",
    "relatedpostid": "RelatedPostId", "linktypeid": "LinkTypeId",
    "count": "Count", "excerptpostid": "ExcerptPostId",
}
STATS_TABLE_MAP = {
    "badges": "badges", "comments": "comments", "posts": "posts",
    "posthistory": "postHistory", "postlinks": "postLinks",
    "tags": "tags", "users": "users", "votes": "votes",
}
STATS_DATE_COLS = {"CreationDate", "Date"}
STATS_REF_EPOCH = int(_time.mktime(_time.strptime("2010-07-19 00:00:00", "%Y-%m-%d %H:%M:%S")))


def convert_stats_query(sql):
    """Convert PRICE STATS query to FactorJoin format."""
    sql = SUBPLAN_RE.sub("", sql).strip()
    sql = re.sub(r"\s+and\s+", " AND ", sql, flags=re.IGNORECASE)
    m = re.search(r"\bWHERE\b", sql, re.IGNORECASE)
    if m:
        prefix = sql[:m.end()]
        body = sql[m.end():]
        body = body.replace("(", "").replace(")", "")
        sql = prefix + body
    sql = sql.rstrip(";").strip() + ";"

    # Split at WHERE
    parts = re.split(r"\bWHERE\b", sql, maxsplit=1, flags=re.IGNORECASE)
    from_part = parts[0]
    where_part = parts[1] if len(parts) > 1 else ""

    # Table + alias replacement in FROM
    for price_tbl, fj_tbl in STATS_TABLE_MAP.items():
        for price_al, fj_al in STATS_ALIAS_MAP.items():
            from_part = re.sub(rf"\b{price_tbl}\s+as\s+{price_al}\b",
                               f"{fj_tbl} as {fj_al}", from_part, flags=re.IGNORECASE)
            from_part = re.sub(rf"\b{price_tbl}\s+{price_al}\b",
                               f"{fj_tbl} as {fj_al}", from_part, flags=re.IGNORECASE)

    # Alias replacement in WHERE
    for price_al, fj_al in STATS_ALIAS_MAP.items():
        where_part = re.sub(rf"\b{price_al}\.", f"{fj_al}.", where_part)

    # Column name case fix
    for lc, cc in STATS_COL_MAP.items():
        where_part = re.sub(rf"\.{lc}\b", f".{cc}", where_part, flags=re.IGNORECASE)

    # Date integer conversion: Unix epoch -> relative to 2010-07-19
    for dcol in STATS_DATE_COLS:
        pattern = rf"(\w+\.{dcol}\s*(?:<=|>=|<|>|=)\s*)(\d+)"
        def replace_ts(m):
            return m.group(1) + str(int(m.group(2)) - STATS_REF_EPOCH)
        where_part = re.sub(pattern, replace_ts, where_part)

    return from_part + "WHERE" + where_part if where_part else from_part


def run_eval(dataset):
    workloads_all = f"{PRICE_DIR}/datas/workloads/test/{dataset}/workloads_all.sql"

    if dataset == "imdb":
        model_path = f"{FJ_DIR}/checkpoints/model_imdb-light_fixed_start_key_200.pkl"
        convert_fn = convert_imdb_query
    else:
        model_path = f"{FJ_DIR}/checkpoints/model_stats_greedy_500.pkl"
        convert_fn = convert_stats_query

    with open(model_path, "rb") as f:
        be = pickle.load(f)
    for t in be.bns:
        be.bns[t].init_inference_method()

    with open(workloads_all) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}, {len(lines)} queries from workloads_all.sql")
    print(f"Model: {model_path}")
    print(f"{'='*60}")

    # warmup
    for ln in lines[:10]:
        sql = ln.split("||")[0].strip()
        try:
            be.get_cardinality_bound_one(convert_fn(sql))
        except:
            pass

    perror_lines = []
    qerrors = []
    latencies = []
    n_single = 0
    n_multi = 0
    n_fail = 0

    for i, ln in enumerate(lines):
        parts = ln.split("||")
        sql_orig = parts[0].strip()
        true_card = int(parts[1])
        tag = parts[2] if len(parts) > 2 else ""

        # Count tables
        fj_sql = convert_fn(sql_orig)
        from_match = re.search(r"\bFROM\b(.*?)\bWHERE\b", fj_sql, re.IGNORECASE)
        if from_match:
            n_tables = from_match.group(1).count(",") + 1
        else:
            from_match2 = re.search(r"\bFROM\b(.*?)$", fj_sql, re.IGNORECASE)
            n_tables = from_match2.group(1).count(",") + 1 if from_match2 else 1

        if n_tables < 2:
            # Single-table: use -1 (true card) for E2E
            perror_lines.append(f"{sql_orig}||{true_card}||-1||{tag}")
            n_single += 1
            # Still compute q-error as 1.0 (perfect estimate)
            qerrors.append(1.0)
            latencies.append(0)
            continue

        n_multi += 1
        t = _time.time()
        try:
            pred = be.get_cardinality_bound_one(fj_sql)
            pred = max(pred, 1)
        except Exception as e:
            pred = 1
            n_fail += 1
        elapsed = _time.time() - t
        latencies.append(elapsed)

        true = max(true_card, 1)
        qerrors.append(max(pred / true, true / pred))
        perror_lines.append(f"{sql_orig}||{true_card}||{pred}||{tag}")

    # Save perror_input
    os.makedirs(f"{EST_DIR}/{dataset}", exist_ok=True)
    out_path = f"{EST_DIR}/{dataset}/factorjoin_perror_input.sql"
    with open(out_path, "w") as f:
        for ln in perror_lines:
            f.write(ln + "\n")

    qe = np.array(qerrors)
    lat_ms = np.array(latencies) * 1000

    print(f"\nTotal: {len(lines)} lines ({n_single} single-table, {n_multi} multi-table, {n_fail} failed)")
    print(f"\n=== Q-errors (all {len(qe)} queries, single-table = 1.0) ===")
    for p in [30, 50, 80, 90, 95, 99]:
        print(f"  p{p}: {np.percentile(qe, p):.4f}")
    print(f"  max:  {qe.max():.4f}")
    print(f"  mean: {qe.mean():.4f}")

    # Multi-table only
    multi_qe = np.array([q for q, l in zip(qerrors, [0]*n_single + [1]*n_multi) if l == 1])
    if len(multi_qe) > 0:
        print(f"\n=== Q-errors (multi-table only, {len(multi_qe)} queries) ===")
        for p in [30, 50, 80, 90, 95, 99]:
            print(f"  p{p}: {np.percentile(multi_qe, p):.4f}")
        print(f"  max:  {multi_qe.max():.4f}")

    multi_lat = lat_ms[lat_ms > 0]
    print(f"\n=== Inference (multi-table) ===")
    print(f"  total: {multi_lat.sum():.1f} ms")
    print(f"  mean:  {multi_lat.mean():.3f} ms/query")
    print(f"  median:{np.median(multi_lat):.3f} ms/query")

    print(f"\nSaved perror_input to: {out_path}")


if __name__ == "__main__":
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["imdb", "stats"]
    for ds in datasets:
        run_eval(ds)
