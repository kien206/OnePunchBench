"""Pull the Open LLM Leaderboard v2 archive (HF dataset open-llm-leaderboard/contents) into
registry.parquet and scores.parquet (source=archive). First step of the data pipeline.

    python scripts/pull_open_llm_leaderboard.py                 # download (or reuse the local snapshot)
    python scripts/pull_open_llm_leaderboard.py --refresh       # force a fresh download
    python scripts/pull_open_llm_leaderboard.py --snapshot data/archive/oll_v2_contents.parquet   # offline

Existing registry rows (matched on model_id, whatever their pinned revision) are only filled where null (enrichment from hf_to_registry /
assign_splits is never overwritten); archive score rows replace older ones by key.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


import onepunch  # noqa: F401,E402  (loads .env: HF_TOKEN, HF_HOME) before huggingface_hub
from onepunch.data.benchmarks import load_benchmarks  # noqa: E402
from onepunch.data.census import ARCHIVE_REPO, build_registry_and_scores  # noqa: E402
from onepunch.data.registry import Registry  # noqa: E402
from onepunch.data.scores import ScoreTable  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", default="data/archive/oll_v2_contents.parquet",
                    help="raw archive snapshot (read if it exists and --refresh is not set, else written)")
    ap.add_argument("--refresh", action="store_true", help="re-download from the Hub")
    ap.add_argument("--registry", default="data/registry/registry.parquet")
    ap.add_argument("--scores", default="data/scores/scores.parquet")
    ap.add_argument("--benchmarks", default="data/benchmarks.yaml")
    args = ap.parse_args(argv)

    snap = Path(args.snapshot)
    if snap.exists() and not args.refresh:
        archive = pd.read_parquet(snap)
    else:
        from datasets import load_dataset

        archive = load_dataset(ARCHIVE_REPO, split="train").to_pandas()
        snap.parent.mkdir(parents=True, exist_ok=True)
        archive.to_parquet(snap, index=False)
    print(f"archive rows: {len(archive)}")

    reg_rows, score_rows = build_registry_and_scores(archive, load_benchmarks(args.benchmarks))

    reg_path = Path(args.registry)
    reg = Registry.load(reg_path) if reg_path.exists() else Registry()
    reg.upsert(reg_rows, prefer="old", by_model=True)
    reg.save(reg_path)

    sc_path = Path(args.scores)
    table = ScoreTable.load(sc_path) if sc_path.exists() else ScoreTable()
    table.append(score_rows)
    table.save(sc_path)

    df = reg.df
    print(f"registry: {len(df)} models, {df['family'].nunique()} families -> {reg_path}")
    print(f"  split: {df['split'].value_counts().to_dict()}")
    print(f"  merged: {int(df['merged'].sum())}  on_hub: {int(df['on_hub'].sum())}  "
          f"tier S: {int((df['tier'] == 'S').sum())}")
    print(f"scores: {len(table)} rows -> {sc_path}")


if __name__ == "__main__":
    main()
