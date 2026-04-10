"""Compute q-errors. Must be imported after all modules are loaded (e.g. via run_experiment imports)."""
import numpy as np


def compute_stats_qerrors(model_path, query_file):
    import pickle
    with open(model_path, 'rb') as f:
        be = pickle.load(f)
    for table in be.bns:
        be.bns[table].init_inference_method()

    with open(query_file, 'r') as f:
        lines = f.readlines()

    qerrors = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split('||')
        true_card = int(parts[0])
        query = parts[1].rstrip(';')
        pred = be.get_cardinality_bound_one(query)
        if pred < 1:
            pred = 1
        if true_card < 1:
            true_card = 1
        qe = max(pred / true_card, true_card / pred)
        qerrors.append(qe)

    qerrors = np.array(qerrors)
    print(f'=== Q-Errors ({len(qerrors)} queries) ===')
    for p in [50, 75, 90, 95, 99, 100]:
        print(f'  {p}th percentile: {np.percentile(qerrors, p):.2f}')
    print(f'  Mean: {np.mean(qerrors):.2f}')
    print(f'  Median: {np.median(qerrors):.2f}')
