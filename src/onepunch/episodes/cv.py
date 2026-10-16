"""Family-grouped cross-validation for the ridge strength lambda0."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np
import torch

from onepunch.episodes.loss import fit_predict, standardize_scales
from onepunch.readout.base import Readout
from onepunch.readout.ridge import RidgeReadout


def family_folds(families: Sequence, max_folds: int = 10, seed: int = 0) -> list[np.ndarray]:
    """Boolean held-out masks: leave-one-family-out, or `max_folds` family groups if more families."""
    fam = np.asarray(families)
    fams = sorted(set(fam.tolist()))
    if len(fams) <= max_folds:
        return [fam == f for f in fams]
    order = np.random.default_rng(seed).permutation(len(fams))
    return [np.isin(fam, [fams[i] for i in order[k::max_folds]]) for k in range(max_folds)]


def select_lambda0(Phi: np.ndarray, S: np.ndarray, families: Sequence, grid: Sequence[float],
                   readout_factory: Callable[..., Readout] = RidgeReadout, sigma=None,
                   max_folds: int = 10) -> tuple[float, dict[float, float]]:
    """Pick lambda0 minimizing held-out-family MSE in standardized units.

    sigma=None -> standardize with the std of S over all rows (pass np.ones(K) for raw units).
    Returns (best_lambda0, {lambda0: mse}).
    """
    Phi_t = torch.as_tensor(np.asarray(Phi), dtype=torch.float64)
    S_np = np.asarray(S, dtype=np.float64)
    S_t = torch.as_tensor(S_np)
    sig = standardize_scales(S_np) if sigma is None else np.asarray(sigma, dtype=np.float64)
    folds = family_folds(families, max_folds)
    table: dict[float, float] = {}
    for lam in grid:
        readout = readout_factory(lambda0=float(lam))
        sse, cnt = 0.0, 0
        for te in folds:
            tr = ~te
            pred = fit_predict(Phi_t[tr], S_t[tr], Phi_t[te], readout, sig).detach().numpy()
            sse += float((((pred - S_np[te]) / sig) ** 2).sum())
            cnt += pred.size
        table[float(lam)] = sse / cnt
    best = min(table, key=lambda k: (table[k], k))
    return best, table
