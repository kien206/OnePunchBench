"""Gate 0 report: compare our scores with the archive; exit 1 if any benchmark fails.

    python scripts/check_reproduction.py [--model-ids ...]
Rule: |ours - archive| <= 2 * sqrt(se_ours^2 + se_archive^2) per model x benchmark.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from onepunch.data.scores import ScoreTable
from onepunch.groundtruth.reproduce import compare, summarize


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", default="data/scores/scores.parquet")
    ap.add_argument("--model-ids", nargs="*")
    ap.add_argument("--z", type=float, default=2.0)
    args = ap.parse_args(argv)

    df = ScoreTable.load(args.scores).df
    if args.model_ids:
        df = df[df["model_id"].isin(args.model_ids)]
    t = compare(df, z=args.z)
    if t.empty:
        print("no model has both 'ours' and 'archive' scores yet")
        return 2
    with pd.option_context("display.width", 200, "display.max_rows", 500, "display.float_format", "{:.4f}".format):
        print(t.assign(passed=t["passed"].map({True: "ok", False: "FAIL"})).to_string(index=False))
    s = summarize(t)
    print(f"\n{s['n_pairs'] - s['n_failed']}/{s['n_pairs']} model x benchmark pairs within {args.z:g} SE "
          f"({s['n_models']} models)")
    return 1 if s["n_failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
