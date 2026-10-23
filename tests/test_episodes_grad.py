import torch

from onepunch.episodes import episode_loss, episode_losses, validation_loss
from onepunch.readout import RidgeReadout
from onepunch.types import Episode

DT = torch.float64


def _problem(n_in=9, n_out=5, d=4, k=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    Phi = torch.randn(n_in + n_out, d, generator=g, dtype=DT)
    S = torch.randn(n_in + n_out, k, generator=g, dtype=DT)
    ep = Episode(torch.arange(n_in).numpy(), torch.arange(n_in, n_in + n_out).numpy())
    return Phi, S, ep


def test_closed_form_hypergradient_matches_autograd():
    for n_in, d in [(9, 4), (4, 9)]:  # primal and dual regimes
        Phi, S, ep = _problem(n_in=n_in, d=d)
        n_out, K = 5, S.shape[1]
        readout = RidgeReadout(0.3)
        lam = 0.3 * n_in
        Phi = Phi.clone().requires_grad_(True)
        loss = episode_loss(Phi, S, [ep], readout, sigma=None, center=False)
        loss.backward()

        with torch.no_grad():
            Pi, Po = Phi[:n_in], Phi[n_in:]
            Si, So = S[:n_in], S[n_in:]
            W = torch.linalg.solve(Pi.T @ Pi + lam * torch.eye(d, dtype=DT), Pi.T @ Si)
            gamma = 2.0 / (n_out * K)
            d_out = Po @ W - So
            d_in = Si - Pi @ W
            Z = torch.linalg.solve(Pi.T @ Pi + lam * torch.eye(d, dtype=DT), Po.T @ d_out)
            g_out = gamma * d_out @ W.T
            g_in = gamma * (d_in @ Z.T - Pi @ Z @ W.T)
        assert torch.allclose(Phi.grad[n_in:], g_out, atol=1e-10)
        assert torch.allclose(Phi.grad[:n_in], g_in, atol=1e-10)


def test_gradcheck_with_centering_and_sigma():
    Phi, S, ep = _problem(n_in=6, n_out=3, d=3)
    ep2 = Episode(torch.tensor([0, 2, 4, 5, 6, 8]).numpy(), torch.tensor([1, 3, 7]).numpy())
    sigma = torch.tensor([0.5, 1.5, 2.0], dtype=DT)
    readout = RidgeReadout(0.2)
    Phi = Phi.clone().requires_grad_(True)
    f = lambda P: episode_loss(P, S, [ep, ep2], readout, sigma, center=True)
    assert torch.autograd.gradcheck(f, (Phi,), eps=1e-6, atol=1e-6)


def test_multi_prompt_matches_loop_and_shapes():
    g = torch.Generator().manual_seed(3)
    Phi = torch.randn(14, 5, 4, generator=g, dtype=DT)  # [N, P, d]
    S = torch.rand(14, 3, generator=g, dtype=DT)
    _, _, ep = _problem()
    eps = [ep, Episode(ep.out_idx, ep.in_idx)]
    sigma = S.std(0)
    r = RidgeReadout(0.1)
    batched = episode_loss(Phi, S, eps, r, sigma)
    assert batched.shape == (5,) and episode_losses(Phi, S, eps, r, sigma).shape == (2, 5)
    for p in range(5):
        assert torch.allclose(batched[p], episode_loss(Phi[:, p], S, eps, r, sigma), atol=1e-12)


def test_validation_loss_matches_manual():
    g = torch.Generator().manual_seed(4)
    Ptr, Pm = torch.randn(12, 4, generator=g, dtype=DT), torch.randn(5, 4, generator=g, dtype=DT)
    Str, Sm = torch.rand(12, 2, generator=g, dtype=DT), torch.rand(5, 2, generator=g, dtype=DT)
    sigma = Str.std(0)
    r = RidgeReadout(0.1)
    got = validation_loss(Ptr, Str, Pm, Sm, r, sigma)
    mu_f, mu_s = Ptr.mean(0), Str.mean(0)
    W = r.fit(Ptr - mu_f, (Str - mu_s) / sigma).W
    want = ((((Pm - mu_f) @ W) - (Sm - mu_s) / sigma) ** 2).mean()
    assert torch.allclose(got, want)


def test_cv_selects_sensible_lambda(population):
    import numpy as np
    from onepunch.episodes import select_lambda0
    rng = np.random.default_rng(0)
    Z = population.latent
    Phi = Z @ rng.normal(size=(3, 8)) + 0.3 * rng.normal(size=(len(Z), 8))
    best, table = select_lambda0(Phi, population.S, population.families, [1e-6, 1e-3, 1e-1, 1.0, 100.0])
    assert best in table and table[best] == min(table.values())
    assert table[100.0] > table[best]  # over-regularized is worse
