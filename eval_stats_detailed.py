"""Detailed FactorJoin STATS eval with warmup and percentile breakdown."""
import time
import numpy as np
import pandas as pd
import os
import run_experiment  # noqa — resolves circular imports
import pickle

model_path = "checkpoints/model_stats_greedy_200.pkl"
query_file = "stats_queries_clean.sql"

with open(model_path, "rb") as f:
    bound_ensemble = pickle.load(f)
for table in bound_ensemble.bns:
    bound_ensemble.bns[table].init_inference_method()

with open(query_file, "r") as f:
    queries = f.readlines()

true_card = pd.read_csv("true_cards_stats.csv")["True cardinality"].values

# warmup
for q in queries[:20]:
    try:
        bound_ensemble.get_cardinality_bound_one(q)
    except Exception:
        pass

q_errors = []
latencies = []
n_fail = 0
for i, query in enumerate(queries):
    t = time.time()
    try:
        pred = bound_ensemble.get_cardinality_bound_one(query)
    except Exception:
        pred = 1
        n_fail += 1
    elapsed = time.time() - t
    latencies.append(elapsed)
    pred = max(pred, 1)
    true = max(true_card[i], 1)
    q_errors.append(max(true / pred, pred / true))

lat_ms = np.array(latencies) * 1000
qe = np.array(q_errors)

print(f"Queries: {len(queries)} ({n_fail} failed)")
print(f"\n=== Inference Time (ms) ===")
print(f"  total: {lat_ms.sum():.2f} ms")
print(f"  mean:  {lat_ms.mean():.3f} ms/query")
print(f"  median:{np.median(lat_ms):.3f} ms/query")
print(f"  p95:   {np.percentile(lat_ms, 95):.3f} ms/query")
print(f"  max:   {lat_ms.max():.3f} ms/query")
print(f"  min:   {lat_ms.min():.3f} ms/query")

print(f"\n=== Q-errors ===")
for p in [30, 50, 80, 90, 95, 99]:
    print(f"  {p}%: {np.percentile(qe, p):.4f}")
print(f"  max:  {qe.max():.4f}")
print(f"  mean: {qe.mean():.4f}")

print(f"\n=== Model Size ===")
print(f"  {os.path.getsize(model_path)} bytes = {os.path.getsize(model_path)/1024/1024:.2f} MB")
