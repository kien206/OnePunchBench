import numpy as np
import torch

from omegaconf import OmegaConf

from onepunch.readout import RidgeReadout, build_readout, lambda0_grid


def _data(n, d, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(n, d, generator=g, dtype=torch.float64),
            torch.randn(n, k, generator=g, dtype=torch.float64))


def _reference(Phi, S, lam):
    Phi, S = Phi.numpy(), S.numpy()
    return np.linalg.inv(Phi.T @ Phi + lam * np.eye(Phi.shape[1])) @ Phi.T @ S


def test_primal_equals_dual_and_reference():
    for n, d in [(20, 6), (6, 20), (10, 10)]:
        Phi, S = _data(n, d)
        Wp = RidgeReadout(0.3, form="primal").fit(Phi, S).W
        Wd = RidgeReadout(0.3, form="dual").fit(Phi, S).W
        Wa = RidgeReadout(0.3, form="auto").fit(Phi, S).W
        assert torch.allclose(Wp, Wd, atol=1e-8)
        assert torch.allclose(Wa, Wp, atol=1e-8)
        np.testing.assert_allclose(Wp.numpy(), _reference(Phi, S, 0.3 * n), atol=1e-8)


def test_scale_with_n_flag_and_dtype():
    Phi, S = _data(8, 4)
    W1 = RidgeReadout(2.0, scale_with_n=False).fit(Phi.float(), S.float()).W
    assert W1.dtype == torch.float64
    np.testing.assert_allclose(W1.numpy(), _reference(Phi.float().double(), S.float().double(), 2.0), atol=1e-8)


def test_gradient_flows_and_predict():
    for n, d in [(12, 4), (4, 12)]:
        Phi, S = _data(n, d)
        Phi.requires_grad_(True)
        f = RidgeReadout(0.1).fit(Phi, S)
        f.predict(Phi).pow(2).sum().backward()
        assert Phi.grad is not None and torch.isfinite(Phi.grad).all() and Phi.grad.abs().sum() > 0
        assert set(f.state_dict()) == {"W"}


def test_batched_matches_loop():
    g = torch.Generator().manual_seed(1)
    Phi = torch.randn(5, 10, 4, generator=g, dtype=torch.float64)
    S = torch.randn(10, 2, generator=g, dtype=torch.float64)
    for form in ("primal", "dual"):
        r = RidgeReadout(0.2, form=form)
        Wb = r.fit(Phi, S).W
        for i in range(5):
            assert torch.allclose(Wb[i], r.fit(Phi[i], S).W, atol=1e-10)


def test_build_readout_from_config():
    cfg = OmegaConf.load("configs/readout/ridge.yaml")
    r = build_readout(cfg)
    assert isinstance(r, RidgeReadout) and r.closed_form
    assert len(lambda0_grid(cfg)) >= 3
