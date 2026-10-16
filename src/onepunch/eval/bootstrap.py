"""Family bootstrap: resample FAMILIES with replacement so big families do not narrow the CI."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

MetricFn = Callable[[np.ndarray, np.ndarray, np.ndarray], float]


def family_bootstrap(metric_fn: MetricFn, pred, true, families: Sequence, n_boot: int = 1000,
                     rng: np.random.Generator | None = None,
                     alpha: float = 0.05) -> tuple[float, float, float]:
    """Return (point estimate, lo, hi) of metric_fn(pred, true, families).

    Each replicate draws as many families as there are, with replacement, and takes ALL models
    of each drawn family; a family drawn twice is relabeled so it counts as two families.
    Non-finite replicates are dropped.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    pred, true, fam = np.asarray(pred), np.asarray(true), np.asarray(families)
    fams = sorted(set(fam.tolist()))
    idx_of = [np.flatnonzero(fam == f) for f in fams]
    point = float(metric_fn(pred, true, fam))
    vals = []
    for _ in range(n_boot):
        draw = rng.integers(len(fams), size=len(fams))
        idx = np.concatenate([idx_of[j] for j in draw])
        lab = np.concatenate([np.full(len(idx_of[j]), r) for r, j in enumerate(draw)])
        vals.append(metric_fn(pred[idx], true[idx], lab))
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return point, float(lo), float(hi)
