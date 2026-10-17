"""Prompt-independent features from model metadata (no forward pass), e.g. the log-parameter baseline."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from onepunch.types import ModelSpec

_COLUMNS = {
    "log_params": lambda s: np.log(s.params_b) if s.params_b else np.nan,
}


class ParamFeatures:
    """FeatureSource whose features ignore the prompt: one value per column per model.

    columns: names in `_COLUMNS` (currently just log_params). The same row is returned for every
    prompt, so a prompt search over these features is pointless; use with search=none.
    """

    def __init__(self, columns: Sequence[str] = ("log_params",)):
        unknown = [c for c in columns if c not in _COLUMNS]
        if unknown:
            raise ValueError(f"unknown column(s) {unknown}; available: {sorted(_COLUMNS)}")
        self.columns = list(columns)
        self.d = len(self.columns)

    @classmethod
    def from_cfg(cls, cfg: Any, persist: bool | None = None, share_with: Any = None) -> "ParamFeatures":
        return cls(list(cfg.features.get("columns", ["log_params"])))

    def features(self, models: Sequence[ModelSpec], prompts: Sequence[bytes]) -> np.ndarray:
        row = np.array([[_COLUMNS[c](m) for c in self.columns] for m in models], dtype=np.float64)
        missing = [m.model_id for m, r in zip(models, row) if not np.isfinite(r).all()]
        if missing:
            raise ValueError(f"no parameter count for {missing[:5]} (not in the registry?)")
        return np.repeat(row[:, None, :], len(prompts), axis=1)
