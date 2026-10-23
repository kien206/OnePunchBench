"""Evaluation metrics. pred/true are [N, K] (or [N]); NaN entries in `true` are ignored."""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np
from scipy.stats import ConstantInputWarning, spearmanr


def _2d(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[:, None] if x.ndim == 1 else x


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0:
        return float("nan")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConstantInputWarning)
        return float(spearmanr(a[ok], b[ok]).statistic)


def spearman_per_benchmark(pred, true) -> np.ndarray:
    p, t = _2d(pred), _2d(true)
    return np.array([_spearman(p[:, k], t[:, k]) for k in range(t.shape[1])])


def mse_per_benchmark(pred, true) -> np.ndarray:
    p, t = _2d(pred), _2d(true)
    with np.errstate(all="ignore"):
        return np.nanmean((p - t) ** 2, axis=0)


def standardized_mse(pred, true, sigma) -> float:
    """mean(((pred - true) / sigma)^2) over models and benchmarks: the units of the search loss."""
    p, t = _2d(pred), _2d(true)
    with np.errstate(all="ignore"):
        return float(np.nanmean(((p - t) / np.asarray(sigma, dtype=np.float64)) ** 2))


def rmse_per_benchmark(pred, true) -> np.ndarray:
    """RMSE in raw score points (scores in [0,1] -> x100)."""
    return 100.0 * np.sqrt(mse_per_benchmark(pred, true))


def within_family_spearman(pred, true, families: Sequence, min_size: int = 4) -> np.ndarray:
    """Spearman within each family with >= min_size models, averaged over families -> [K]."""
    p, t = _2d(pred), _2d(true)
    fam = np.asarray(families)
    out = np.full(t.shape[1], np.nan)
    groups = [np.flatnonzero(fam == f) for f in sorted(set(fam.tolist()))]
    groups = [g for g in groups if len(g) >= min_size]
    for k in range(t.shape[1]):
        vals = [_spearman(p[g, k], t[g, k]) for g in groups]
        if np.isfinite(vals).any():
            out[k] = np.nanmean(vals)
    return out


def _nanmean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


# Scalar metrics with the (pred, true, families) signature used by family_bootstrap.
def mean_spearman(pred, true, families=None) -> float:
    return _nanmean(spearman_per_benchmark(pred, true))


def mean_rmse(pred, true, families=None) -> float:
    return _nanmean(rmse_per_benchmark(pred, true))


def mean_within_spearman(pred, true, families, min_size: int = 4) -> float:
    return _nanmean(within_family_spearman(pred, true, families, min_size))


def summarize(pred, true, families: Sequence, benchmarks: Sequence[str], min_size: int = 4) -> dict[str, float]:
    """Flat dict: `<metric>/<benchmark>` and `<metric>/mean` for spearman, rmse, mse, within_spearman."""
    per = {
        "spearman": spearman_per_benchmark(pred, true),
        "rmse": rmse_per_benchmark(pred, true),
        "mse": mse_per_benchmark(pred, true),
        "within_spearman": within_family_spearman(pred, true, families, min_size),
    }
    out: dict[str, float] = {}
    for name, vals in per.items():
        for b, v in zip(benchmarks, vals):
            out[f"{name}/{b}"] = float(v)
        out[f"{name}/mean"] = _nanmean(vals)
    return out
