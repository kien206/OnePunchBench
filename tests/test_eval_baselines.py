from types import SimpleNamespace

import numpy as np
import pytest

from onepunch.eval import baseline_metrics, standardized_mse
from onepunch.readout import RidgeReadout
from onepunch.types import ModelSpec

GRID = [1e-3, 1e-2, 1e-1]


def _pop(params_b, families, S):
    specs = [ModelSpec(model_id=f"m{i}", params_b=float(p)) for i, p in enumerate(params_b)]
    return SimpleNamespace(specs=specs, S=np.asarray(S, dtype=float), families=np.asarray(families))


def _linear_in_log_params(n, seed):
    rng = np.random.default_rng(seed)
    params = rng.uniform(0.1, 3.0, size=n)
    S = 0.3 + 0.05 * np.log(params)[:, None] * np.array([1.0, 0.5])
    return _pop(params, [f"fam{i % 5}" for i in range(n)], S)


def test_mean_baseline_on_train_is_one(population):
    out = baseline_metrics(["mean"], population, {"train": population}, lambda: RidgeReadout(0.1), GRID)
    assert np.isclose(out["baseline/mean/train/loss"], 1.0)  # standardized by the train sigma itself
    assert np.isnan(out["baseline/mean/train/spearman"])     # constant prediction has no ranking
    assert out["baseline/mean/train/lofo_loss"] > 1.0       # held-out families sit away from the others' mean


def test_log_params_baseline_recovers_size_law():
    tr, te = _linear_in_log_params(40, 0), _linear_in_log_params(15, 1)
    out = baseline_metrics(["mean", "log_params"], tr, {"testA": te}, lambda: RidgeReadout(0.1), GRID)
    assert out["baseline/log_params/testA/loss"] < 0.01 < out["baseline/mean/testA/loss"]
    assert out["baseline/log_params/testA/spearman"] > 0.99
    assert out["baseline/log_params/train/lofo_loss"] < out["baseline/mean/train/lofo_loss"]


def test_unknown_baseline():
    with pytest.raises(ValueError, match="unknown baseline"):
        baseline_metrics(["oracle"], _linear_in_log_params(10, 0), {}, lambda: RidgeReadout(0.1), GRID)


def test_standardized_mse():
    assert np.isclose(standardized_mse([[0.5, 0.2]], [[0.3, 0.1]], [0.1, 0.1]), (4 + 1) / 2)
