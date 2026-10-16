"""Open LLM Leaderboard v2 archive -> registry rows + long-format score rows.

Pure functions on DataFrames (no network); scripts/pull_open_llm_leaderboard.py does the IO.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from onepunch.data.benchmarks import Benchmark, sampling_se
from onepunch.types import REGISTRY_COLUMNS, SCORE_COLUMNS

ARCHIVE_REPO = "open-llm-leaderboard/contents"
PROTOCOL = "oll_v2"
TIER_S_MAX_B = 4.0
_NO_BASE = {"", "removed", "?", "unknown", "none", "nan"}
_UNKNOWN_ARCH = {"", "?", "unknown"}


def norm_id(model_id: str) -> str:
    """Normalized HF id used for matching and as family name (lowercase, stripped)."""
    return model_id.strip().lower()


def clean_base_model(raw, model_id: str = "") -> str | None:
    """Parent id from the archive's `Base Model`, or None.

    None for: missing / "Removed" / "?" values, the model itself (a root), and
    "org/name (Merge)" markers (the leaderboard's "base unknown").
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    s = str(raw).strip()
    if s.lower() in _NO_BASE or re.search(r"\(merge\)\s*$", s, flags=re.I):
        return None
    if " " in s or "," in s:  # not a single repo id
        return None
    return None if norm_id(s) == norm_id(model_id) else s


def resolve_roots(
    parents: Mapping[str, Sequence[str]], merged: Iterable[str] = ()
) -> dict[str, str | None]:
    """Root model of every key of `parents`, following parent links (case-insensitive).

    - no parent -> the model is its own root;
    - one parent -> root of that parent;
    - several parents (or a model in `merged`) -> the common root if all parents
      share one, else None; a merged model without known parents -> None;
    - cycles are cut deterministically: the root is the smallest id on the cycle.
    The returned root keeps the spelling it has in `parents` (key spelling first).
    """
    low = norm_id
    canon: dict[str, str] = {}
    plist: dict[str, list[str]] = {}
    for m, ps in parents.items():
        canon.setdefault(low(m), m)
    for m, ps in parents.items():
        plist[low(m)] = [low(p) for p in ps if low(p) != low(m)]
        for p in ps:
            canon.setdefault(low(p), p)
    merged_l = {low(m) for m in merged}
    memo: dict[str, str | None] = {}

    def root(m: str, path: tuple[str, ...]) -> str | None:
        if m in memo:
            return memo[m]
        if m in path:
            return min(path[path.index(m):])
        ps = plist.get(m, [])
        if not ps:
            r = None if m in merged_l else m
        else:
            rs = {root(p, path + (m,)) for p in ps}
            r = next(iter(rs)) if len(rs) == 1 else None
        memo[m] = r
        return r

    out = {}
    for m in parents:
        r = root(low(m), ())
        out[m] = canon[r] if r is not None else None
    return out


def _unit_scale(col: pd.Series) -> pd.Series:
    """Raw leaderboard columns are in [0,1]; convert from percent if a column is not."""
    col = pd.to_numeric(col, errors="coerce")
    return col / 100.0 if col.max() > 1.0 + 1e-9 else col


def build_registry_and_scores(
    archive: pd.DataFrame,
    benchmarks: Mapping[str, Benchmark],
    parents_extra: Mapping[str, Sequence[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map the archive (open-llm-leaderboard/contents, train) to registry and score rows.

    Archive columns used: fullname (model id), Base Model, Merged, MoE,
    Architecture, #Params (B), Precision, Chat Template, Available on the hub,
    Upload To Hub Date / Submission Date, Weight type, plus the raw score
    column of every benchmark. parents_extra maps merged models to parents
    known from elsewhere (Hub card). A merge (Merged flag or merge type) whose
    parents are unknown or do not share one root gets split="excluded". Duplicate fullnames keep the latest submission.
    """
    a = archive.sort_values("Submission Date", kind="stable").drop_duplicates("fullname", keep="last")
    a = a.sort_values("fullname", kind="stable").reset_index(drop=True)
    extra = parents_extra or {}

    ids = a["fullname"].tolist()
    # merged: archive flag or merge-type row. The "(Merge)" marker in Base Model
    # only says "base unknown" (it also appears on plain fine-tunes), so it does
    # not make a row merged; such rows get no parent and stay their own root
    # until hf_to_registry fills `parents` from the Hub card.
    merged_flag = (a["Merged"].astype(bool) | a["Type"].astype(str).str.contains("merges")).to_numpy()
    parents = {}
    for mid, base, mg in zip(ids, a["Base Model"], merged_flag):
        b = clean_base_model(base, mid)
        if mid in extra:
            parents[mid] = list(extra[mid])
        else:
            parents[mid] = [b] if b else []
    roots = resolve_roots(parents, merged={m for m, f in zip(ids, merged_flag) if f})

    rows = []
    for (_, r), mg in zip(a.iterrows(), merged_flag):
        mid = r["fullname"]
        root = roots[mid]
        params = r["#Params (B)"]
        arch = str(r["Architecture"]).strip()
        upload = str(r.get("Upload To Hub Date", "") or "").strip()
        precision = str(r["Precision"])
        loader = "hf"
        if r["Weight type"] != "Original":
            loader = "unsupported_adapter"
        elif precision not in ("bfloat16", "float16", "float32"):
            loader = "unsupported_" + precision
        rows.append({
            "model_id": mid,
            "revision": "main",
            "family": norm_id(root if root else mid),
            "root_model": root,
            "parents": parents[mid],
            "tokenizer_hash": None,
            "vocab_size": None,
            "params_b": float(params) if params and params > 0 else np.nan,
            "arch": None if arch.lower() in _UNKNOWN_ARCH else arch,
            "moe": bool(r["MoE"]),
            "merged": bool(mg),
            "release_date": upload or str(r["Submission Date"]),
            "dtype": precision if precision in ("bfloat16", "float16", "float32") else "bfloat16",
            "loader": loader,
            "trust_remote_code": False,
            "chat_template": bool(r["Chat Template"]),
            "on_hub": bool(r["Available on the hub"]),
            "tier": "S" if (params and 0 < params <= TIER_S_MAX_B) else "R",
            "split": "excluded" if (root is None or loader != "hf") else "train",
            "surrogate_group": None,
        })
    registry = pd.DataFrame(rows, columns=list(REGISTRY_COLUMNS))

    parts = []
    for name, b in benchmarks.items():
        s = _unit_scale(a[b.archive_raw_column])
        p = pd.DataFrame({
            "model_id": a["fullname"],
            "revision": "main",
            "benchmark": name,
            "score": s,
            "stderr": sampling_se(s.fillna(0.0), b.n_items),
            "n_items": b.n_items,
            "source": "archive",
            "protocol_version": PROTOCOL,
            "date": a["Submission Date"].astype(str),
        })
        parts.append(p[s.notna()])
    scores = pd.concat(parts, ignore_index=True)[list(SCORE_COLUMNS)]
    return registry, scores
