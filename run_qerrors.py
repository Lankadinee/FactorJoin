"""Wrapper to compute q-errors, triggering all module loads first."""
# These imports resolve the circular import by loading everything upfront
from Join_scheme.data_prepare import convert_time_to_int
from Evaluation.training import train_one_stats, train_one_imdb
from Evaluation.testing import test_on_stats, test_on_imdb, test_on_imdb_light
from compute_qerrors import compute_stats_qerrors

import sys

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python run_qerrors.py <model_path> <query_file>")
        print("  query_file format: true_card||SQL_query")
        sys.exit(1)

    compute_stats_qerrors(sys.argv[1], sys.argv[2])
