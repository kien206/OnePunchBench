"""Prompt-free reference points, reported next to the prompt's metrics in every training run.

    mean        predict the training-set mean score for every model
    log_params  ridge on log(parameter count) (features.params.ParamFeatures)

A prompt is only useful if its features beat these on held-out families (`train/lofo_loss`)
and on the evaluation splits. Losses are in the units of the search loss: squared error
divided by the training sigma, averaged over models and benchmarks.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

from onepunch.episodes import family_folds, select_lambda0, standardize_scales
from onepunch.eval.lofo import fit_predictor
from onepunch.eval.metrics import mean_rmse, mean_spearman, standardized_mse
from onepunch.features.params import ParamFeatures
from onepunch.readout.base import Readout

BASELINES = ("mean", "log_params")


def _mean_lofo_loss(S: np.ndarray, families: np.ndarray) -> float:
    """Held-out-family loss of predicting the mean of the other families (same folds as select_lambda0)."""
    sig = standardize_scales(S)
    sse, cnt = 0.0, 0
    for te in family_folds(families):
        err = (S[te] - S[~te].mean(0)) / sig
        sse, cnt = sse + float((err ** 2).sum()), cnt + err.size
    return sse / cnt


def baseline_metrics(names: Sequence[str], train: Any, splits: Mapping[str, Any],
                     make_readout: Callable[[], Readout], grid: Sequence[float],
                     standardize: bool = True) -> dict[str, float]:
    """Flat dict `baseline/<name>/train/lofo_loss` and `baseline/<name>/<split>/{loss,spearman,rmse}`.

    train / splits values: populations with `.specs`, `.S`, `.families` (utils.experiment.Population).
    make_readout: a fresh readout of the run's config; its lambda0 is chosen by family-grouped CV.
    """
    unknown = [n for n in names if n not in BASELINES]
    if unknown:
        raise ValueError(f"unknown baseline(s) {unknown}; available: {list(BASELINES)}")
    sigma = standardize_scales(train.S, standardize)
    out: dict[str, float] = {}
    for name in names:
        if name == "mean":
            s_bar = train.S.mean(0)
            predict = lambda p, s_bar=s_bar: np.tile(s_bar, (len(p.specs), 1))  # noqa: E731
            lofo = _mean_lofo_loss(train.S, train.families)
        else:
            src = ParamFeatures(["log_params"])
            phi = lambda p, src=src: src.features(p.specs, [b""])[:, 0]  # noqa: E731
            lam, table = select_lambda0(phi(train), train.S, train.families, grid)
            readout = make_readout()
            if hasattr(readout, "lambda0"):
                readout.lambda0 = lam
            fitted = fit_predictor(phi(train), train.S, readout, standardize)
            predict = lambda p, f=fitted, phi=phi: f.predict(phi(p))  # noqa: E731
            lofo = table[lam]
        out[f"baseline/{name}/train/lofo_loss"] = lofo
        for split, pop in splits.items():
            pred = predict(pop)
            out[f"baseline/{name}/{split}/loss"] = standardized_mse(pred, pop.S, sigma)
            out[f"baseline/{name}/{split}/spearman"] = mean_spearman(pred, pop.S)
            out[f"baseline/{name}/{split}/rmse"] = mean_rmse(pred, pop.S)
    return out
