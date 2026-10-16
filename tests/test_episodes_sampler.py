import warnings

import numpy as np
import pytest

from onepunch.episodes import EpisodeSampler, sample_step_pool


def _pop(n_fam=12, per_fam=6, n_tok=3):
    fam = np.repeat([f"f{i}" for i in range(n_fam)], per_fam)
    tok = np.repeat([f"t{i % n_tok}" for i in range(n_fam)], per_fam)
    return fam, tok


def _check_basic(eps, fam, sampler):
    for e in eps:
        assert not (set(fam[e.in_idx]) & set(fam[e.out_idx]))
        assert not (set(e.in_idx) & set(e.out_idx))
        for idx in (e.in_idx, e.out_idx):
            assert len(set(idx)) == len(idx)
            _, counts = np.unique(fam[idx], return_counts=True)
            assert counts.max() <= sampler.n_fam_cap


def test_disjoint_cap_sizes():
    fam, tok = _pop(n_fam=16)
    s = EpisodeSampler(n_in=16, n_out=8, n_fam_cap=4, rho=0.0, n_episodes=20)
    eps = s.sample(fam, tok, np.random.default_rng(0))
    assert len(eps) == 20
    _check_basic(eps, fam, s)
    assert all(len(e.in_idx) == 16 and len(e.out_idx) == 8 for e in eps)


def test_tokenizer_disjoint_episodes():
    fam, tok = _pop(n_fam=12)
    s = EpisodeSampler(12, 6, 3, rho=1.0, n_episodes=30)
    eps = s.sample(fam, tok, np.random.default_rng(1), active_tokenizer="t0")
    _check_basic(eps, fam, s)
    for e in eps:
        assert e.tokenizer_disjoint
        assert len(set(tok[e.out_idx])) == 1 and tok[e.out_idx][0] != "t0"
        assert tok[e.out_idx][0] not in set(tok[e.in_idx])


def test_rho_fraction():
    fam, tok = _pop(n_fam=12)
    s = EpisodeSampler(12, 6, 3, rho=0.5, n_episodes=200)
    eps = s.sample(fam, tok, np.random.default_rng(2))
    assert 0.35 < np.mean([e.tokenizer_disjoint for e in eps]) < 0.65


def test_active_group_placement():
    fam, tok = _pop(n_fam=12)
    active = tok == "t0"
    act_fams = set(fam[active])
    s = EpisodeSampler(12, 6, 3, rho=0.0, n_episodes=10)
    eps = s.sample(fam, tok, np.random.default_rng(3), active, "t0")
    _check_basic(eps, fam, s)
    n_in_side = 0
    for e in eps:
        in_f, out_f = set(fam[e.in_idx]), set(fam[e.out_idx])
        in_has, out_has = active[e.in_idx].any(), active[e.out_idx].any()
        assert in_has != out_has  # exactly one side holds the active group, at least one model
        if in_has:
            n_in_side += 1
            assert not (act_fams & out_f)
        else:
            assert not (act_fams & in_f)
    assert n_in_side == 5  # half inner, half outer

    s2 = EpisodeSampler(12, 6, 3, rho=1.0, n_episodes=20)
    for e in s2.sample(fam, tok, np.random.default_rng(4), active, "t0"):
        assert e.tokenizer_disjoint and active[e.in_idx].any() and not active[e.out_idx].any()


def test_determinism():
    fam, tok = _pop()
    s = EpisodeSampler(12, 6, 3, rho=0.3, n_episodes=6)
    a = s.sample(fam, tok, np.random.default_rng(7), tok == "t1", "t1")
    b = s.sample(fam, tok, np.random.default_rng(7), tok == "t1", "t1")
    for x, y in zip(a, b):
        assert (x.in_idx == y.in_idx).all() and (x.out_idx == y.out_idx).all()
        assert x.tokenizer_disjoint == y.tokenizer_disjoint


def test_infeasible_sizes_warn_once_no_crash():
    fam, tok = _pop(n_fam=4, per_fam=2, n_tok=2)
    s = EpisodeSampler(32, 16, 4, rho=0.5, n_episodes=10)
    with pytest.warns(UserWarning):
        eps = s.sample(fam, tok, np.random.default_rng(0), tok == "t0", "t0")
    _check_basic(eps, fam, s)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        s.sample(fam, tok, np.random.default_rng(1))  # no second warning
    # single tokenizer + rho=1 falls back to standard episodes
    s1 = EpisodeSampler(4, 2, 2, rho=1.0, n_episodes=3)
    f1, t1 = _pop(n_fam=6, n_tok=1)
    with pytest.warns(UserWarning):
        eps = s1.sample(f1, t1, np.random.default_rng(0))
    assert len(eps) == 3 and not any(e.tokenizer_disjoint for e in eps)
    # one family only: nothing feasible, but no crash
    s2 = EpisodeSampler(4, 2, 2, rho=0.0, n_episodes=3)
    with pytest.warns(UserWarning):
        assert s2.sample(np.array(["a"] * 5), np.array(["t"] * 5), np.random.default_rng(0)) == []


def test_step_pool():
    fam, tok = _pop(n_fam=12)
    pool = sample_step_pool(fam, np.random.default_rng(0), n_families=5, models_per_family=3)
    assert len(pool) == 15 and len(set(fam[pool])) == 5
    assert all((fam[pool] == f).sum() == 3 for f in set(fam[pool]))
    mask = tok == "t2"
    for seed in range(10):
        p = sample_step_pool(fam, np.random.default_rng(seed), 3, 2, eligible_mask=mask, must_include_mask=fam == "f5")
        assert mask[p].all() and "f5" in set(fam[p])
    p = sample_step_pool(fam, np.random.default_rng(0), 3, 2)
    assert (p == sample_step_pool(fam, np.random.default_rng(0), 3, 2)).all()
