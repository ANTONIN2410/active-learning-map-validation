"""Run the full benchmark (16 cases) and save the results to reports/results/.

    python scripts/run_benchmark.py --replicates 1000 --workers 8
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from alval.benchmark import headline, run_case  # noqa: E402
from alval.data import PRODUCTS, REGIONS, load_points  # noqa: E402


def _case(args):
    region, product, n_rep = args
    return run_case(load_points(), region, product, n_rep=n_rep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    jobs = [(r, p, args.replicates) for r in REGIONS for p in PRODUCTS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        out = list(pool.map(_case, jobs))

    results = pd.DataFrame([row for rows, _ in out for row in rows])
    models = pd.DataFrame([m for _, m in out])
    dest = ROOT / "reports" / "results"
    dest.mkdir(parents=True, exist_ok=True)
    results.to_csv(dest / "benchmark.csv", index=False)
    models.to_csv(dest / "error_models.csv", index=False)
    print((100 * headline(results)).round(1).to_string())


if __name__ == "__main__":
    main()
