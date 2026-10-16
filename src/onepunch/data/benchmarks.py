"""Benchmark definitions (data/benchmarks.yaml) and sampling-noise helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DEFAULT_PATH = "data/benchmarks.yaml"


@dataclass(frozen=True)
class Benchmark:
    name: str
    lm_eval_task: str
    metrics: tuple[str, ...]  # lm-eval metric names; averaged if several
    n_shots: int
    n_items: int
    archive_raw_column: str


def load_benchmarks(path: str | Path = DEFAULT_PATH) -> dict[str, Benchmark]:
    """Return {name: Benchmark} in file order."""
    cfg = yaml.safe_load(Path(path).read_text())["benchmarks"]
    return {
        name: Benchmark(
            name=name,
            lm_eval_task=c["lm_eval_task"],
            metrics=tuple(c["metrics"]),
            n_shots=int(c["n_shots"]),
            n_items=int(c["n_items"]),
            archive_raw_column=c["archive_raw_column"],
        )
        for name, c in cfg.items()
    }


def sampling_se(p, n):
    """Binomial standard error sqrt(p(1-p)/n); broadcasts over arrays."""
    p = np.clip(np.asarray(p, dtype=np.float64), 0.0, 1.0)
    return np.sqrt(p * (1.0 - p) / np.asarray(n, dtype=np.float64))


def noise_ceiling_rmse(S: np.ndarray, n_items) -> np.ndarray:
    """Per-benchmark RMSE that pure test-set sampling noise puts on a target.

    S: [N, K] scores (or [K] vector), n_items: [K] (or scalar). Returns [K]:
    sqrt(mean_i p_ik (1 - p_ik) / n_k). Even a perfect predictor of the latent
    ability cannot beat this against observed scores.
    """
    S = np.atleast_2d(np.asarray(S, dtype=np.float64))
    var = sampling_se(S, np.asarray(n_items, dtype=np.float64)) ** 2
    return np.sqrt(np.nanmean(var, axis=0))
