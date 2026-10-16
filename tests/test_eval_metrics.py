import numpy as np
from scipy.stats import spearmanr

from onepunch.eval import (family_bootstrap, mean_spearman, mean_within_spearman, summarize,
                           spearman_per_benchmark)


def _data(seed=0, n=40, k=3):
    rng = np.random.default_rng(seed)
    true = rng.random((n, k))
    pred = true + 0.2 * rng.normal(size=(n, k))
    fam = np.repeat([f"f{i}" for i in range(n // 5)], 5)
    return pred, true, fam


def test_metrics_vs_scipy():
    pred, true, fam = _data()
    b = ["a", "b", "c"]
    out = summarize(pred, true, fam, b)
    for k, name in enumerate(b):
        assert np.isclose(out[f"spearman/{name}"], spearmanr(pred[:, k], true[:, k]).statistic)
        assert np.isclose(out[f"mse/{name}"], np.mean((pred[:, k] - true[:, k]) ** 2))
        assert np.isclose(out[f"rmse/{name}"], 100 * np.sqrt(np.mean((pred[:, k] - true[:, k]) ** 2)))
        w = [spearmanr(pred[fam == f, k], true[fam == f, k]).statistic for f in sorted(set(fam))]
        assert np.isclose(out[f"within_spearman/{name}"], np.mean(w))
    assert np.isclose(out["spearman/mean"], np.mean([out[f"spearman/{n}"] for n in b]))


def test_within_family_min_size_and_nan_safety():
    pred, true, fam = _data()
    assert np.isnan(summarize(pred, true, fam, list("abc"), min_size=6)["within_spearman/mean"])
    true2 = true.copy()
    true2[:7, 0] = np.nan
    pred2 = pred.copy()
    pred2[:, 1] = 1.0  # constant prediction -> NaN spearman, other benchmarks unaffected
    out = summarize(pred2, true2, fam, list("abc"))
    assert np.isnan(out["spearman/b"]) and np.isfinite(out["spearman/a"]) and np.isfinite(out["spearman/mean"])
    assert np.isclose(out["mse/a"], np.nanmean((pred2[:, 0] - true2[:, 0]) ** 2))


def test_bootstrap_resamples_families():
    # One huge family dominates the models; the metric is computed per family so a
    # model-level bootstrap would give a narrow CI, a family bootstrap a wide one.
    rng = np.random.default_rng(0)
    fam = np.array(["big"] * 100 + [f"s{i}" for i in range(4)] * 2)
    true = rng.random((len(fam), 1))
    pred = np.where((fam == "big")[:, None], true + 0.01 * rng.normal(size=true.shape), rng.random(true.shape))
    metric = lambda p, t, f: float(np.mean(np.abs(p - t)))
    pt, lo, hi = family_bootstrap(metric, pred, true, fam, n_boot=500, rng=np.random.default_rng(1))
    assert lo <= pt <= hi
    # same draws as a manual family resample
    fams = sorted(set(fam))
    r = np.random.default_rng(1)
    vals = []
    for _ in range(500):
        d = r.integers(len(fams), size=len(fams))
        idx = np.concatenate([np.flatnonzero(fam == fams[j]) for j in d])
        vals.append(metric(pred[idx], true[idx], None))
    assert np.isclose(lo, np.quantile(vals, 0.025)) and np.isclose(hi, np.quantile(vals, 0.975))
    assert hi - lo > 0.05  # wide: whole families come and go together


def test_bootstrap_duplicate_families_relabeled():
    seen = []

    def metric(p, t, f):
        seen.append(len(set(f.tolist())))
        return 0.0
    fam = np.array(["a", "a", "b", "b", "c", "c"])
    x = np.arange(6.0)[:, None]
    family_bootstrap(metric, x, x, fam, n_boot=20, rng=np.random.default_rng(0))
    assert all(n == 3 for n in seen[1:])  # drawn-twice families count as two families
