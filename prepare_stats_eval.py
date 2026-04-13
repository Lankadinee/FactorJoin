"""Convert PRICE STATS-CEB queries to FactorJoin format.

PRICE uses: lowercase columns, st_X aliases, Unix epoch integers for dates
FactorJoin uses: CamelCase columns, short aliases (b/c/u/ph/v/pl/t/p), integer dates relative to 2010-07-19

Since we trained on CSVs preprocessed with convert_time_to_int (relative to
2010-07-19 local time), query date integers must use the same base.
PRICE's Unix epoch = FactorJoin's relative + EPOCH_OFFSET.
"""
import re
import csv
import time as _time

PRICE_WORKLOAD = "/home/student.unimelb.edu.au/lrathuwadu/PRICE/datas/workloads/test/stats/workloads.sql"
SUBPLAN_COMMENT_RE = re.compile(r"/\*\s*\(.*?\)\s*\*/")

# FactorJoin's reference epoch
REF_EPOCH = int(_time.mktime(_time.strptime("2010-07-19 00:00:00", "%Y-%m-%d %H:%M:%S")))

# PRICE alias -> FactorJoin alias
ALIAS_MAP = {
    "st_b": "b", "st_c": "c", "st_p": "p", "st_ph": "ph",
    "st_pl": "pl", "st_t": "t", "st_u": "u", "st_v": "v",
}

# PRICE lowercase col -> FactorJoin CamelCase col
COL_MAP = {
    "id": "Id", "userid": "UserId", "postid": "PostId",
    "creationdate": "CreationDate", "date": "Date",
    "score": "Score", "votetypeid": "VoteTypeId",
    "bountyamount": "BountyAmount", "posthistorytypeid": "PostHistoryTypeId",
    "posttypeid": "PostTypeId", "viewcount": "ViewCount",
    "owneruserid": "OwnerUserId", "answercount": "AnswerCount",
    "commentcount": "CommentCount", "favoritecount": "FavoriteCount",
    "lasteditoruserid": "LastEditorUserId", "reputation": "Reputation",
    "views": "Views", "upvotes": "UpVotes", "downvotes": "DownVotes",
    "relatedpostid": "RelatedPostId", "linktypeid": "LinkTypeId",
    "count": "Count", "excerptpostid": "ExcerptPostId",
}

# Table names PRICE -> FactorJoin (PRICE uses lowercase in FROM)
TABLE_MAP = {
    "badges": "badges", "comments": "comments", "posts": "posts",
    "posthistory": "postHistory", "postlinks": "postLinks",
    "tags": "tags", "users": "users", "votes": "votes",
}

# Date columns that need epoch conversion
DATE_COLS = {"CreationDate", "Date"}


def convert_query(sql):
    """Convert a PRICE STATS SQL to FactorJoin format."""
    # Strip comment prefix
    sql = SUBPLAN_COMMENT_RE.sub("", sql).strip()
    # Normalize AND
    sql = re.sub(r"\s+and\s+", " AND ", sql, flags=re.IGNORECASE)
    # Strip parens from WHERE conditions
    where_m = re.search(r"\bWHERE\b", sql, re.IGNORECASE)
    if where_m:
        prefix = sql[:where_m.end()]
        body = sql[where_m.end():]
        body = body.replace("(", "").replace(")", "")
        sql = prefix + body
    sql = sql.rstrip(";").strip() + ";"

    # Replace table names in FROM clause (before WHERE)
    parts = re.split(r"\bWHERE\b", sql, maxsplit=1, flags=re.IGNORECASE)
    from_part = parts[0]
    where_part = parts[1] if len(parts) > 1 else ""

    # Replace "tablename as st_x" or "tablename st_x" with correct case
    for price_tbl, fj_tbl in TABLE_MAP.items():
        # "tablename as st_x" -> "fj_tbl as fj_alias"
        for price_al, fj_al in ALIAS_MAP.items():
            from_part = re.sub(
                rf"\b{price_tbl}\s+as\s+{price_al}\b",
                f"{fj_tbl} as {fj_al}",
                from_part, flags=re.IGNORECASE
            )
            from_part = re.sub(
                rf"\b{price_tbl}\s+{price_al}\b",
                f"{fj_tbl} as {fj_al}",
                from_part, flags=re.IGNORECASE
            )

    # In WHERE: replace alias.col references
    for price_al, fj_al in ALIAS_MAP.items():
        # Replace alias prefix
        where_part = re.sub(rf"\b{price_al}\.", f"{fj_al}.", where_part)

    # Replace column names (after alias dot)
    for price_col, fj_col in COL_MAP.items():
        where_part = re.sub(rf"\.{price_col}\b", f".{fj_col}", where_part, flags=re.IGNORECASE)

    # Convert date integer values: PRICE Unix epoch -> FJ relative
    # Find patterns like "alias.DateCol <= 1409892677" and convert the int
    for dcol in DATE_COLS:
        pattern = rf"(\w+\.{dcol}\s*(?:<=|>=|<|>|=)\s*)(\d+)"
        def replace_ts(m):
            prefix = m.group(1)
            unix_ts = int(m.group(2))
            fj_ts = unix_ts - REF_EPOCH
            return f"{prefix}{fj_ts}"
        where_part = re.sub(pattern, replace_ts, where_part)

    sql_out = from_part + "WHERE" + where_part
    return sql_out


with open(PRICE_WORKLOAD) as f:
    lines = [ln.strip() for ln in f if ln.strip()]

queries = []
true_cards = []

for ln in lines:
    parts = ln.split("||")
    if len(parts) < 3:
        continue
    sql = parts[0].strip()
    true_card = int(parts[1])
    converted = convert_query(sql)
    queries.append(converted)
    true_cards.append(true_card)

# Write in test_on_stats format: TRUE_CARD||SQL (for FactorJoin's eval)
with open("stats_queries_fj.sql", "w") as f:
    for tc, q in zip(true_cards, queries):
        f.write(f"{tc}||{q}\n")

# Also write separate files for custom eval
with open("stats_queries_clean.sql", "w") as f:
    for q in queries:
        f.write(q + "\n")

with open("true_cards_stats.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["True cardinality"])
    for tc in true_cards:
        w.writerow([tc])

print(f"wrote {len(queries)} queries")
print(f"sample: {queries[0][:120]}...")
print(f"sample: {queries[50][:120]}...")
