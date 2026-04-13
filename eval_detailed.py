"""Detailed FactorJoin eval — must be run via: uv run python eval_detailed.py"""
import time
import numpy as np
import pandas as pd
import os

# Force correct import order to avoid circular import
import run_experiment  # noqa — resolves Join_scheme circular deps
import pickle

model_path = "checkpoints/model_imdb-light_fixed_start_key_200.pkl"
query_file = "joblight_queries.sql"

with open(model_path, "rb") as f:
    bound_ensemble = pickle.load(f)
for table in bound_ensemble.bns:
    bound_ensemble.bns[table].init_inference_method()

with open(query_file, "r") as f:
    queries = f.readlines()

true_card = pd.read_csv("true_cards_joblightranges.csv")["True cardinality"].values

# warmup
for q in queries[:20]:
    bound_ensemble.get_cardinality_bound_one(q)

q_errors = []
latencies = []
for i, query in enumerate(queries):
    t = time.time()
    pred = bound_ensemble.get_cardinality_bound_one(query)
    elapsed = time.time() - t
    latencies.append(elapsed)
    pred = max(pred, 1)
    true = max(true_card[i], 1)
    q_errors.append(max(true / pred, pred / true))

lat_ms = np.array(latencies) * 1000
qe = np.array(q_errors)

print(f"Queries: {len(queries)}")
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
print(f"  {os.path.getsize(model_path)} bytes = {os.path.getsize(model_path)/1024:.1f} KB")
