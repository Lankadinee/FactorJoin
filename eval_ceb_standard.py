"""Evaluate FactorJoin on the standard CEB sub-plan queries (2603 queries).

1. Run predictions via FactorJoin
2. Execute each sub-query against Postgres for true cardinalities
3. Compute q-errors
"""
import run_experiment  # noqa — circular import fix
import pickle
import time
import numpy as np
import psycopg2

MODEL = "checkpoints/model_stats_greedy_200.pkl"
QUERY_FILE = "/tmp/stats_CEB_sub_queries.sql"
DB_KWARGS = dict(dbname="stats", user="postgres", password="postgres",
                 host="localhost", port=5456)

# Load model
with open(MODEL, "rb") as f:
    be = pickle.load(f)
for t in be.bns:
    be.bns[t].init_inference_method()

# Load queries
with open(QUERY_FILE) as f:
    lines = [ln.strip() for ln in f if ln.strip()]

queries = []
for ln in lines:
    parts = ln.split("||")
    sql = parts[0].strip()
    queries.append(sql)

print(f"Loaded {len(queries)} sub-plan queries")

# Step 1: FactorJoin predictions + timing
# Warmup
for q in queries[:20]:
    try:
        be.get_cardinality_bound_one(q)
    except Exception:
        pass

preds = []
latencies = []
n_fail = 0
for i, q in enumerate(queries):
    t = time.time()
    try:
        pred = be.get_cardinality_bound_one(q)
    except Exception:
        pred = 1
        n_fail += 1
    latencies.append(time.time() - t)
    preds.append(max(pred, 1))
    if (i + 1) % 500 == 0:
        print(f"  predicted {i+1}/{len(queries)}")

print(f"Predictions done: {len(preds)} ({n_fail} failed)")
lat_ms = np.array(latencies) * 1000
print(f"Inference: total {lat_ms.sum():.1f} ms, mean {lat_ms.mean():.3f} ms, median {np.median(lat_ms):.3f} ms")

# Step 2: True cardinalities from Postgres
print("\nExecuting queries against Postgres for true cardinalities...")
conn = psycopg2.connect(**DB_KWARGS)
conn.set_session(autocommit=True)
cur = conn.cursor()
cur.execute("SET statement_timeout = '300s';")

true_cards = []
for i, q in enumerate(queries):
    try:
        cur.execute(q)
        row = cur.fetchone()
        true_cards.append(max(int(row[0]), 1))
    except Exception as e:
        print(f"  query {i} failed: {e}")
        true_cards.append(1)
        conn.rollback()
        cur = conn.cursor()
        cur.execute("SET statement_timeout = '300s';")
    if (i + 1) % 500 == 0:
        print(f"  executed {i+1}/{len(queries)}")

conn.close()
print(f"True cards collected: {len(true_cards)}")

# Step 3: Q-errors
qe = []
for p, t in zip(preds, true_cards):
    qe.append(max(p / t, t / p))

qe = np.array(qe)
print(f"\n=== Standard CEB Sub-Plan Q-errors ({len(qe)} queries) ===")
for p in [30, 50, 80, 90, 95, 99]:
    print(f"  p{p}: {np.percentile(qe, p):.4f}")
print(f"  max:  {qe.max():.4f}")
print(f"  mean: {qe.mean():.4f}")

# Breakdown by join size
from collections import Counter
sizes = []
for q in queries:
    n_tables = q.upper().split(" WHERE ")[0].count(",") + 1
    sizes.append(n_tables)

print(f"\n=== By join size ===")
for sz in sorted(set(sizes)):
    idxs = [i for i, s in enumerate(sizes) if s == sz]
    sub_qe = qe[idxs]
    print(f"  {sz}-table ({len(idxs)}): p50={np.percentile(sub_qe, 50):.2f}, "
          f"p90={np.percentile(sub_qe, 90):.2f}, p99={np.percentile(sub_qe, 99):.2f}")
