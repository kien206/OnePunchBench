"""Deterministic family -> split assignment (see data/splits.yaml)."""

from __future__ import annotations

import re
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from onepunch.data.census import norm_id, resolve_roots
from onepunch.types import SPLITS

LISTED_SPLITS = ("val", "calib", "testA")


def load_splits_config(path: str | Path = "data/splits.yaml") -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text()) or {}


def _match(name: str | None, patterns: Sequence[str], ci: bool = False) -> bool:
    if not name:
        return False
    if ci:
        return any(fnmatchcase(name.lower(), p.lower()) for p in patterns)
    return any(fnmatchcase(name, p) for p in patterns)


def _truthy(x) -> bool:
    return x is not None and x is not pd.NA and bool(x)


def canonical_family(root: str, aliases: Mapping[str, str] | None = None) -> str:
    """Lowercased root id, replaced by the first matching alias pattern."""
    f = norm_id(root)
    for pat, name in (aliases or {}).items():
        if fnmatchcase(f, pat.lower()):
            return name
    return f


def recompute_roots(df: pd.DataFrame, extra_parents: Mapping[str, Sequence[str]] | None = None) -> pd.DataFrame:
    """Return df with root_model recomputed from the `parents` links.

    Links come from the registry rows plus `extra_parents` (Hub cache; used for
    models not in the registry, and for registry models without parents). A row
    with no parents but a root_model different from itself keeps that root.
    """
    extra = dict(extra_parents or {})
    pm: dict[str, list[str]] = {k: list(v) for k, v in extra.items()}
    for mid, parents, root in zip(df["model_id"], df["parents"], df["root_model"]):
        if isinstance(parents, (list, tuple, np.ndarray)) and len(parents):
            pm[mid] = list(parents)
        elif mid not in pm or not pm[mid]:
            keep = root if isinstance(root, str) and norm_id(root) != norm_id(mid) else None
            pm[mid] = [keep] if keep else list(pm.get(mid, []))
    merged = set(df.loc[df["merged"].fillna(False).astype(bool), "model_id"])
    roots = resolve_roots(pm, merged=merged)
    out = df.copy()
    out["root_model"] = [roots.get(m) for m in out["model_id"]]
    return out


def assign_splits(
    df: pd.DataFrame, cfg: Mapping[str, Any], extra_parents: Mapping[str, Sequence[str]] | None = None
) -> pd.DataFrame:
    """Set root_model, family, split and surrogate_group on a registry DataFrame.

    Rules, in order (the whole family always gets one split):
      0. per-model `overrides` win; unscorable models (loader "unsupported*",
         excluded_archs) and merges without a common root -> "excluded"
         (their family is the model itself);
      1. family first released after cutoff_date -> testB;
      2. family whose majority of models is MoE / hybrid / recurrent -> testC;
      3. family matches a val / calib / testA pattern -> that split;
      4. families with >= train_min_models hub models -> train, else "excluded".
    Models of a testC-free family that are individually MoE/hybrid -> "excluded".
    """
    df = recompute_roots(df, extra_parents)
    aliases = cfg.get("family_aliases") or {}
    fam = [canonical_family(r if isinstance(r, str) else m, aliases) for r, m in zip(df["root_model"], df["model_id"])]
    df["family"] = fam
    df = df.reset_index(drop=True)

    excl_archs = cfg.get("excluded_archs") or []
    unscorable = np.array([
        (isinstance(l, str) and l.startswith("unsupported")) or _match(a, excl_archs)
        for l, a in zip(df["loader"], df["arch"])
    ]) | df["root_model"].isna().to_numpy()

    c_archs = (cfg.get("testC") or {}).get("archs") or []
    use_moe = (cfg.get("testC") or {}).get("use_moe_flag", True)
    special = np.array([
        (use_moe and _truthy(m)) or _match(a, c_archs)
        for m, a in zip(df["moe"], df["arch"])
    ])

    ok = ~unscorable
    g = pd.DataFrame({"family": df["family"], "date": df["release_date"], "special": special,
                      "hub": (df["on_hub"] != False).fillna(True).astype(bool)})[ok]  # noqa: E712
    stats = g.groupby("family").agg(first=("date", lambda x: x.dropna().min() if x.notna().any() else None),
                                    frac_special=("special", "mean"), hub=("hub", "sum"))
    cutoff = str(cfg.get("cutoff_date", "2025-03-13"))
    min_n = int(cfg.get("train_min_models", 3))

    def family_split(f: str, row) -> str:
        if row["first"] is not None and str(row["first"]) > cutoff:
            return "testB"
        if row["frac_special"] > 0.5:
            return "testC"
        for s in LISTED_SPLITS:
            if _match(f, cfg.get(s) or [], ci=True):
                return s
        return "train" if row["hub"] >= min_n else "excluded"

    fsplit = {f: family_split(f, r) for f, r in stats.iterrows()}
    split = []
    for f, bad, sp in zip(df["family"], unscorable, special):
        if bad:
            split.append("excluded")
        else:
            s = fsplit[f]
            split.append("excluded" if (sp and s != "testC") else s)
    for mid, s in (cfg.get("overrides") or {}).items():
        if s not in SPLITS:
            raise ValueError(f"override split {s!r} for {mid} not in {SPLITS}")
        split = [s if m == mid else x for m, x in zip(df["model_id"], split)]
    df["split"] = split

    df["surrogate_group"] = surrogate_groups(df, cfg.get("surrogates") or {})
    return df


def surrogate_groups(df: pd.DataFrame, surrogates: Mapping[str, Sequence[str]]) -> list[str | None]:
    """Group name per row: train models whose tokenizer_hash equals a surrogate's."""
    hash_to_group: dict[str, str] = {}
    by_id = dict(zip(df["model_id"], df["tokenizer_hash"]))
    for name, refs in surrogates.items():
        for ref in refs:
            h = by_id.get(ref)
            if isinstance(h, str) and h:
                hash_to_group.setdefault(h, name)
                break
    return [
        hash_to_group.get(h) if (s == "train" and isinstance(h, str)) else None
        for h, s in zip(df["tokenizer_hash"], df["split"])
    ]
