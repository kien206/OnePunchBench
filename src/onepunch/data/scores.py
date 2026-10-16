"""Long-format score table (data/scores/scores.parquet)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from onepunch.types import SCORE_COLUMNS

KEY = ["model_id", "revision", "benchmark", "source"]
SOURCES = ("archive", "ours")


class ScoreTable:
    """Raw scores in [0,1], one row per (model_id, revision, benchmark, source)."""

    def __init__(self, df: pd.DataFrame | None = None):
        if df is None:
            df = pd.DataFrame({c: [] for c in SCORE_COLUMNS})
        missing = [c for c in SCORE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"score columns missing: {missing}")
        df = df[list(SCORE_COLUMNS)].copy()
        df["score"] = df["score"].astype("float64")
        df["stderr"] = df["stderr"].astype("float64")
        df["n_items"] = pd.to_numeric(df["n_items"], errors="coerce").astype("Int64")
        self.df = self._dedupe(df)
        self.validate()

    @staticmethod
    def _dedupe(df: pd.DataFrame) -> pd.DataFrame:
        # newest date wins; ties resolved by row order (later row wins)
        df = df.assign(_pos=np.arange(len(df)))
        df = df.sort_values(["date", "_pos"], kind="stable", na_position="first")
        df = df.drop_duplicates(KEY, keep="last").drop(columns="_pos")
        return df.sort_values(KEY, kind="stable").reset_index(drop=True)

    # ---- io ----
    @classmethod
    def load(cls, path: str | Path) -> "ScoreTable":
        return cls(pd.read_parquet(path))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(path, index=False)

    def __len__(self) -> int:
        return len(self.df)

    # ---- editing ----
    def validate(self) -> None:
        d = self.df
        if d["score"].isna().any() or ((d["score"] < 0) | (d["score"] > 1)).any():
            raise ValueError("scores must be non-null and in [0, 1] (raw accuracy, not percent)")
        bad = set(d["source"]) - set(SOURCES)
        if bad:
            raise ValueError(f"unknown source(s): {bad}")
        if d[["model_id", "benchmark", "protocol_version"]].isna().any().any():
            raise ValueError("model_id, benchmark and protocol_version must be set")

    def append(self, rows: pd.DataFrame) -> "ScoreTable":
        """Add rows; on duplicate (model_id, revision, benchmark, source) the newest date wins."""
        new = ScoreTable(pd.concat([self.df, ScoreTable(rows).df], ignore_index=True))
        self.df = new.df
        return self

    # ---- views ----
    def matrix(
        self,
        model_ids: Sequence[str],
        benchmarks: Sequence[str],
        sources: Sequence[str] = ("ours", "archive"),
        protocol_version: str = "oll_v2",
    ) -> tuple[np.ndarray, np.ndarray]:
        """Scores as S [N, K] float64 (NaN where missing) and mask [N, K] bool.

        For each (model, benchmark) the first source in `sources` that has a
        value wins. Revision is ignored (newest date wins among revisions of
        one source). Only rows of `protocol_version` are used.
        """
        d = self.df[(self.df["protocol_version"] == protocol_version) & self.df["source"].isin(sources)]
        d = d[d["model_id"].isin(set(model_ids)) & d["benchmark"].isin(set(benchmarks))]
        prio = {s: i for i, s in enumerate(sources)}
        d = d.assign(_prio=d["source"].map(prio)).sort_values(["_prio", "date"], ascending=[True, False], kind="stable")
        best = d.drop_duplicates(["model_id", "benchmark"], keep="first")
        lookup = {(m, b): s for m, b, s in zip(best["model_id"], best["benchmark"], best["score"])}
        S = np.full((len(model_ids), len(benchmarks)), np.nan)
        for i, m in enumerate(model_ids):
            for k, b in enumerate(benchmarks):
                S[i, k] = lookup.get((m, b), np.nan)
        return S, ~np.isnan(S)
