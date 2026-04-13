"""Standard CEB eval with separate query files for FJ (relative dates) and PG (epoch dates)."""
import run_experiment  # noqa
import pickle
import time
import numpy as np
import psycopg2

MODEL = "checkpoints/model_stats_greedy_200.pkl"
FJ_QUERIES = "/tmp/stats_CEB_sub_queries_fj.sql"    # relative dates for model
PG_QUERIES = "/tmp/stats_CEB_sub_queries_int.sql"    # unix epoch for postgres
DB_KWARGS = dict(dbname="stats", user="postgres", password="postgres",
                 host="localhost", port=5456)

with open(MODEL, "rb") as f:
    be = pickle.load(f)
for t in be.bns:
    be.bns[t].init_inference_method()

with open(FJ_QUERIES) as f:
    fj_lines = [ln.strip().split("||")[0].strip() for ln in f if ln.strip()]
with open(PG_QUERIES) as f:
    pg_lines = [ln.strip().split("||")[0].strip() for ln in f if ln.strip()]

print(f"Loaded {len(fj_lines)} queries")

# Warmup
for q in fj_lines[:20]:
    try: be.get_cardinality_bound_one(q)
    except: pass

# Predictions
preds, latencies, n_fail = [], [], 0
for i, q in enumerate(fj_lines):
    t = time.time()
    try:
        pred = be.get_cardinality_bound_one(q)
    except:
        pred = 1; n_fail += 1
    latencies.append(time.time() - t)
    preds.append(max(pred, 1))
    if (i+1) % 500 == 0: print(f"  pred {i+1}/{len(fj_lines)}")

lat_ms = np.array(latencies) * 1000
print(f"Predictions: {len(preds)} ({n_fail} failed)")
print(f"Inference: total {lat_ms.sum():.1f} ms, mean {lat_ms.mean():.3f} ms, median {np.median(lat_ms):.3f} ms")

# True cards from PG
print("\nGetting true cardinalities from Postgres...")
conn = psycopg2.connect(**DB_KWARGS)
conn.set_session(autocommit=True)
cur = conn.cursor()
cur.execute("SET statement_timeout = '300s';")

true_cards = []
for i, q in enumerate(pg_lines):
    try:
        cur.execute(q)
        true_cards.append(max(int(cur.fetchone()[0]), 1))
    except Exception as e:
        if (i+1) % 100 == 0 or i < 5: print(f"  q{i} failed: {str(e)[:80]}")
        true_cards.append(1)
        conn = psycopg2.connect(**DB_KWARGS)
        conn.set_session(autocommit=True)
        cur = conn.cursor()
        cur.execute("SET statement_timeout = '300s';")
    if (i+1) % 500 == 0: print(f"  executed {i+1}/{len(pg_lines)}")
conn.close()

n_failed_pg = sum(1 for t in true_cards if t == 1)
print(f"True cards: {len(true_cards)} ({n_failed_pg} defaulted to 1)")

# Q-errors
qe = np.array([max(p/t, t/p) for p, t in zip(preds, true_cards)])
print(f"\n=== Standard CEB Sub-Plan Q-errors ({len(qe)} queries) ===")
for p in [30, 50, 80, 90, 95, 99]:
    print(f"  p{p}: {np.percentile(qe, p):.4f}")
print(f"  max:  {qe.max():.4f}")
print(f"  mean: {qe.mean():.4f}")

# By join size
sizes = [q.upper().split(" WHERE ")[0].count(",") + 1 for q in pg_lines]
print(f"\n=== By join size ===")
for sz in sorted(set(sizes)):
    idxs = [i for i, s in enumerate(sizes) if s == sz]
    sub_qe = qe[idxs]
    print(f"  {sz}-table ({len(idxs)}): p50={np.percentile(sub_qe, 50):.2f}, "
          f"p90={np.percentile(sub_qe, 90):.2f}, p99={np.percentile(sub_qe, 99):.2f}")
