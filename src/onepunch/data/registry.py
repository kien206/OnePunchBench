"""Model registry: one row per (model_id, revision), stored as parquet."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from onepunch.types import REGISTRY_COLUMNS, ModelSpec

KEY = ["model_id", "revision"]
_BOOL = ("moe", "merged", "trust_remote_code", "chat_template", "on_hub")
_STR = ("model_id", "revision", "family", "root_model", "tokenizer_hash", "arch",
        "release_date", "dtype", "loader", "tier", "split", "surrogate_group")


def _is_null(v: Any) -> bool:
    if isinstance(v, (list, tuple, np.ndarray)):
        return len(v) == 0
    return v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA


def _coerce(df: pd.DataFrame) -> pd.DataFrame:
    """Validate columns and normalize dtypes (nullable bool/int so parquet round-trips)."""
    missing = [c for c in REGISTRY_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REGISTRY_COLUMNS]
    if missing or extra:
        raise ValueError(f"registry columns mismatch: missing={missing} extra={extra}")
    df = df[list(REGISTRY_COLUMNS)].copy()
    for c in _BOOL:
        df[c] = df[c].astype("boolean")
    for c in _STR:
        df[c] = df[c].astype(object).where(df[c].notna(), None)
    df["vocab_size"] = pd.to_numeric(df["vocab_size"], errors="coerce").astype("Int64")
    df["params_b"] = pd.to_numeric(df["params_b"], errors="coerce").astype("float64")
    df["parents"] = [list(p) if isinstance(p, (list, tuple, np.ndarray)) else [] for p in df["parents"]]
    if df.duplicated(KEY).any():
        raise ValueError("duplicate (model_id, revision) rows in registry")
    return df.reset_index(drop=True)


def _as_list(x) -> list | None:
    if x is None:
        return None
    return [x] if isinstance(x, str) else list(x)


class Registry:
    """Typed wrapper around a DataFrame with REGISTRY_COLUMNS."""

    def __init__(self, df: pd.DataFrame | None = None):
        if df is None:
            df = pd.DataFrame({c: [] for c in REGISTRY_COLUMNS})
        self.df = _coerce(df)

    # ---- io ----
    @classmethod
    def load(cls, path: str | Path) -> "Registry":
        return cls(pd.read_parquet(path))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(path, index=False)

    def __len__(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"Registry({len(self)} rows, {self.df['family'].nunique()} families)"

    # ---- editing ----
    def upsert(self, rows: pd.DataFrame, prefer: str = "new", by_model: bool = False) -> "Registry":
        """Insert rows or update existing ones, keyed by (model_id, revision).

        Null values never overwrite. prefer="new": non-null new values replace
        old ones; prefer="old": new values only fill null cells. by_model=True
        matches existing rows on model_id alone and lets the new revision
        replace the old one (used when pinning "main" to a commit sha).
        Mutates and returns self.
        """
        if prefer not in ("new", "old"):
            raise ValueError("prefer must be 'new' or 'old'")
        rows = _coerce(rows)
        cur = self.df.to_dict("records")
        key = (lambda r: r["model_id"]) if by_model else (lambda r: (r["model_id"], r["revision"]))
        index = {key(r): i for i, r in enumerate(cur)}
        for new in rows.to_dict("records"):
            i = index.get(key(new))
            if i is None:
                index[key(new)] = len(cur)
                cur.append(new)
                continue
            old = cur[i]
            for c in REGISTRY_COLUMNS:
                if _is_null(new[c]):
                    continue
                if prefer == "new" or _is_null(old[c]):
                    old[c] = new[c]
        self.df = _coerce(pd.DataFrame(cur, columns=list(REGISTRY_COLUMNS)))
        return self

    # ---- selection ----
    def filter(
        self,
        *,
        split: str | Sequence[str] | None = None,
        tier: str | Sequence[str] | None = None,
        min_params_b: float | None = None,
        max_params_b: float | None = None,
        families: Sequence[str] | None = None,
        exclude_families: Sequence[str] | None = None,
        tokenizer_hash: str | Sequence[str] | None = None,
        on_hub: bool | None = None,
        archs: Sequence[str] | None = None,
        moe: bool | None = None,
        merged: bool | None = None,
        surrogate_group: str | Sequence[str] | None = None,
        model_ids: Sequence[str] | None = None,
        max_per_family: int | None = None,
        seed: int = 0,
    ) -> "Registry":
        """Return a sub-registry (row order preserved). None means "no constraint".

        Rows with unknown (null) values never satisfy a constraint on that column.
        max_per_family is applied last; within a family rows are ranked by a
        seeded hash of model_id@revision, so the choice is deterministic and
        independent of row order.
        """
        df = self.df
        m = np.ones(len(df), dtype=bool)

        def isin(col: str, vals) -> np.ndarray:
            return df[col].isin(_as_list(vals)).to_numpy(dtype=bool)

        if split is not None:
            m &= isin("split", split)
        if tier is not None:
            m &= isin("tier", tier)
        if families is not None:
            m &= isin("family", families)
        if exclude_families is not None:
            m &= ~isin("family", exclude_families)
        if tokenizer_hash is not None:
            m &= isin("tokenizer_hash", tokenizer_hash)
        if archs is not None:
            m &= isin("arch", archs)
        if surrogate_group is not None:
            m &= isin("surrogate_group", surrogate_group)
        if model_ids is not None:
            m &= isin("model_id", model_ids)
        if min_params_b is not None:
            m &= (df["params_b"] >= min_params_b).to_numpy(dtype=bool)
        if max_params_b is not None:
            m &= (df["params_b"] <= max_params_b).to_numpy(dtype=bool)
        for col, val in (("on_hub", on_hub), ("moe", moe), ("merged", merged)):
            if val is not None:
                m &= (df[col] == bool(val)).fillna(False).to_numpy(dtype=bool)
        sub = df[m]
        if max_per_family is not None:
            rank = sub.apply(lambda r: hashlib.sha256(
                f"{seed}|{r['model_id']}@{r['revision']}".encode()).hexdigest(), axis=1)
            order = rank.groupby(sub["family"], dropna=False).rank(method="first")
            sub = sub[(order <= max_per_family).to_numpy()]
        return Registry(sub)

    # ---- accessors ----
    def spec(self, model_id: str) -> ModelSpec:
        rows = self.df[self.df["model_id"] == model_id]
        if rows.empty:
            raise KeyError(model_id)
        return _row_to_spec(rows.iloc[0])

    def specs(self) -> list[ModelSpec]:
        return [_row_to_spec(r) for _, r in self.df.iterrows()]

    def families(self) -> np.ndarray:
        return self.df["family"].to_numpy(dtype=object)

    def model_ids(self) -> list[str]:
        return self.df["model_id"].tolist()


def _row_to_spec(r: pd.Series) -> ModelSpec:
    def get(c, default=None):
        v = r[c]
        return default if _is_null(v) else v

    params = get("params_b")
    return ModelSpec(
        model_id=r["model_id"],
        revision=get("revision", "main"),
        dtype=get("dtype", "bfloat16"),
        loader=get("loader", "hf"),
        trust_remote_code=bool(get("trust_remote_code", False)),
        params_b=None if params is None else float(params),
        family=get("family"),
        tokenizer_hash=get("tokenizer_hash"),
    )


POOL_KEYS = ("description", "model_list", "model_ids", "where")


def read_model_list(path: str | Path) -> list[str]:
    """HF ids from a data/models/*.yaml file (`models: [{model_id: ...}, ...]`)."""
    import yaml

    entries = (yaml.safe_load(Path(path).read_text()) or {}).get("models") or []
    return [e["model_id"] if isinstance(e, Mapping) else str(e) for e in entries]


def select_pool(registry: Registry, pool_cfg: Mapping[str, Any], root: str | Path | None = None) -> Registry:
    """The models of a configs/pool/*.yaml: which models are loaded at all.

    Keys (all optional; a model must satisfy every one that is given):
      model_list  path (or list of paths) to data/models/*.yaml, relative to `root` (default: repo root)
      model_ids   inline list of HF ids
      where       conditions on registry columns = keyword arguments of Registry.filter
                  (tier, max_params_b, on_hub, families, max_per_family, ...)
      description free text
    Which of these models train / monitor / evaluate is NOT decided here (see `roles` in config.yaml).
    """
    try:
        from omegaconf import DictConfig, OmegaConf

        if isinstance(pool_cfg, DictConfig):
            pool_cfg = OmegaConf.to_container(pool_cfg, resolve=True)
    except ImportError:  # pragma: no cover
        pass
    unknown = set(pool_cfg) - set(POOL_KEYS)
    if unknown:
        raise ValueError(f"unknown pool key(s) {sorted(unknown)}; allowed: {POOL_KEYS}")
    if root is None:
        from onepunch.utils.env import REPO_ROOT as root
    kw = dict(pool_cfg.get("where") or {})
    ids: list[str] | None = None
    lists = pool_cfg.get("model_list")
    if lists:
        ids = [m for f in ([lists] if isinstance(lists, (str, Path)) else lists)
               for m in read_model_list(Path(root) / f)]
    if pool_cfg.get("model_ids"):
        inline = list(pool_cfg["model_ids"])
        ids = inline if ids is None else [m for m in ids if m in set(inline)]
    if ids is not None:
        kw["model_ids"] = ids if "model_ids" not in kw else [m for m in kw["model_ids"] if m in set(ids)]
    return registry.filter(**kw)
