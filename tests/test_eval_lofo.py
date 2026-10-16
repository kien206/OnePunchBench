import numpy as np

from onepunch.eval import FittedPredictor, fit_predictor, lofo_predict, mean_spearman
from onepunch.readout import RidgeReadout

PROMPT = [b"a fixed probe prompt"]


def _phi(fake_source, population):
    return fake_source.features(population.specs, PROMPT)[:, 0, :]


def test_lofo_synthetic_high_spearman(fake_source, population):
    Phi = _phi(fake_source, population)
    pred = lofo_predict(Phi, population.S, population.families, RidgeReadout(1e-2))
    assert pred.shape == population.S.shape and pred.min() >= 0 and pred.max() <= 1
    rho = mean_spearman(pred, population.S)
    print(f"LOFO mean Spearman (fake features): {rho:.3f}")
    assert rho > 0.8


def test_lofo_random_features_is_chance(population):
    rng = np.random.default_rng(0)
    rhos = []
    for _ in range(5):
        Phi = rng.normal(size=(len(population.S), 8))
        pred = lofo_predict(Phi, population.S, population.families, RidgeReadout(1e-1))
        rhos.append(mean_spearman(pred, population.S))
    assert abs(np.mean(rhos)) < 0.3


def test_lofo_holds_out_family(fake_source, population):
    # A held-out family's prediction must not depend on its own scores.
    Phi = _phi(fake_source, population)
    S2 = population.S.copy()
    mask = population.families == "fam0"
    S2[mask] = 1 - S2[mask]
    a = lofo_predict(Phi, population.S, population.families, RidgeReadout(1e-2))
    b = lofo_predict(Phi, S2, population.families, RidgeReadout(1e-2))
    np.testing.assert_allclose(a[mask], b[mask], atol=1e-10)


def test_extra_columns(fake_source, population):
    Phi = _phi(fake_source, population)
    logp = np.log([s.params_b for s in population.specs])[:, None]
    pred = lofo_predict(np.hstack([Phi, logp]), population.S, population.families, RidgeReadout(1e-2))
    assert pred.shape == population.S.shape


def test_predictor_roundtrip_and_clipping(tmp_path, fake_source, population):
    Phi = _phi(fake_source, population)
    pr = fit_predictor(Phi, population.S, RidgeReadout(1e-2), meta={"prompt": "hello", "d": 8})
    p1 = pr.predict(Phi)
    assert p1.shape == population.S.shape and p1.min() >= 0 and p1.max() <= 1
    assert np.abs(p1 - population.S).mean() < 0.1  # in-sample fit is decent
    pr.save(tmp_path / "pred.npz")
    pr2 = FittedPredictor.load(tmp_path / "pred.npz")
    np.testing.assert_allclose(pr2.predict(Phi), p1)
    assert pr2.meta == {"prompt": "hello", "d": 8}
    far = Phi.mean(0) + 1e3
    assert pr.predict(far[None]).max() <= 1.0 and pr.predict(-far[None]).min() >= 0.0
    assert np.abs(pr.predict(far[None], clip=False)).max() > 1.0
