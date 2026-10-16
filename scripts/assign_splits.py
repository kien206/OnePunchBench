"""Assign family, split and surrogate_group to every registry row (deterministic).

    python scripts/assign_splits.py                      # uses data/splits.yaml
Re-run after ingest_archive.py / hf_to_registry.py; the result depends only on the
registry content, data/splits.yaml and data/registry/hub_parents.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from onepunch.data.registry import Registry
from onepunch.data.splits import assign_splits, load_splits_config


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default="data/registry/registry.parquet")
    ap.add_argument("--splits", default="data/splits.yaml")
    ap.add_argument("--hub-parents", default="data/registry/hub_parents.json",
                    help="parent links gathered by hf_to_registry (optional)")
    args = ap.parse_args(argv)

    reg = Registry.load(args.registry)
    hp = Path(args.hub_parents)
    extra = json.loads(hp.read_text()) if hp.exists() else {}
    out = Registry(assign_splits(reg.df, load_splits_config(args.splits), extra))
    out.save(args.registry)

    df = out.df
    print(f"{len(df)} models, {df['family'].nunique()} families")
    print("models per split:", df["split"].value_counts().to_dict())
    print("families per split:", df.groupby("split")["family"].nunique().to_dict())
    live = df[(df["split"] != "excluded") & (df["on_hub"] != False)]  # noqa: E712
    print("on-hub models per split:", live["split"].value_counts().to_dict())
    top = live.groupby(["split", "family"]).size().sort_values(ascending=False).head(15)
    print("largest on-hub families:\n" + top.to_string())
    tok = df["tokenizer_hash"].dropna()
    if len(tok):
        print("models per tokenizer hash:", tok.value_counts().to_dict())
        print("surrogate groups:", df["surrogate_group"].value_counts().to_dict())


if __name__ == "__main__":
    main()
