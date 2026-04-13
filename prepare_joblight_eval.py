"""Prepare FactorJoin evaluation files from PRICE's JOB-light workload.

Creates:
1. joblight_queries.sql — one SQL per line (766 sub-queries)
2. true_cards_joblightranges.csv — with 'True cardinality' column
"""
import re
import csv

PRICE_WORKLOAD = "/home/student.unimelb.edu.au/lrathuwadu/PRICE/datas/workloads/test/imdb/workloads.sql"
SUBPLAN_COMMENT_RE = re.compile(r"/\*\s*\(.*?\)\s*\*/")

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
    # strip /* ... */ comment prefix
    sql_clean = SUBPLAN_COMMENT_RE.sub("", sql).strip()
    # normalize AND to uppercase (FactorJoin parser needs uppercase)
    sql_clean = re.sub(r"\s+and\s+", " AND ", sql_clean, flags=re.IGNORECASE)
    # strip parens around WHERE conditions — FactorJoin's parser chokes on them
    where_m = re.search(r"\bWHERE\b", sql_clean, re.IGNORECASE)
    if where_m:
        prefix = sql_clean[:where_m.end()]
        body = sql_clean[where_m.end():]
        body = body.replace("(", "").replace(")", "")
        sql_clean = prefix + body + ";"
    queries.append(sql_clean)
    true_cards.append(true_card)

with open("joblight_queries.sql", "w") as f:
    for q in queries:
        f.write(q + "\n")

with open("true_cards_joblightranges.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["True cardinality"])
    for tc in true_cards:
        w.writerow([tc])

print(f"wrote {len(queries)} queries to joblight_queries.sql")
print(f"wrote {len(true_cards)} true cards to true_cards_joblightranges.csv")
