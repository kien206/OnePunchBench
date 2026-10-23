import math

import numpy as np
import pytest

from onepunch.episodes import EpisodeSampler, standardize_scales
from onepunch.readout import RidgeReadout
from onepunch.search import (BestOfN, EpisodicObjective, PromptConstraints, RandomSwap, SearchLoop,
                             ValidationObjective)
from onepunch.types import Optimizer


class ByteTokenizer:
    """Fake tokenizer: one token per byte (ids 0..255), id 256 = special."""

    all_special_ids = [256]

    def __len__(self):
        return 257

    def encode(self, text, add_special_tokens=False):
        return list(text.encode("utf-8"))

    def decode(self, ids, clean_up_tokenization_spaces=False):
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


class FixedProposals:
    def __init__(self, proposals):
        self.proposals, self.seen_states = proposals, []

    def propose(self, state):
        self.seen_states.append(state)
        return list(self.proposals)


class FuncObjective:
    """Deterministic objective from a function; records every evaluated prompt."""

    def __init__(self, fn):
        self.fn, self.evaluated = fn, []
        self.steps = 0

    def new_step(self, rng, active_mask=None, active_tokenizer=None):
        self.steps += 1

    def __call__(self, prompts):
        self.evaluated.extend(prompts)
        return np.array([self.fn(p) for p in prompts])


def _loss_of(p: bytes) -> float:
    return (sum(p) % 997) / 997.0


def _objective(population, fake_source, **kw):
    sigma = standardize_scales(population.S)
    sampler = EpisodeSampler(n_in=16, n_out=8, n_fam_cap=4, rho=0.25, n_episodes=4)
    return EpisodicObjective(fake_source, population.specs, population.S, population.families,
                             population.tokenizers, sampler, RidgeReadout(0.1), sigma, **kw)


INIT = "The quick brown fox jumps over the lazy dog. Café über naïve.".encode()


def test_step_loss_never_increases(population, fake_source):
    obj = _objective(population, fake_source, step_pool={"n_families": 6, "models_per_family": 4, "redraw_every": 3})
    opt = RandomSwap(ByteTokenizer(), n_candidates=8, allowed="printable_ascii")
    res = SearchLoop(opt, obj, steps=8, val_every=100).run(INIT, np.random.default_rng(0))
    assert len(res.history) == 8
    for h in res.history:
        assert h["loss"] <= h["loss_before"] + 1e-15  # current prompt is among the scored candidates
        assert h["n_valid"] > 0
    assert any(h["loss"] < h["loss_before"] for h in res.history)  # some step improved
    assert math.isnan(res.best_val_loss) and res.best_step_loss == min(h["loss"] for h in res.history)


def test_monotone_across_steps_with_fixed_episodes(population, fake_source):
    obj = _objective(population, fake_source)

    class FixedEpisodes:
        def __init__(self):
            self.drawn = False

        def new_step(self, rng, *a):
            if not self.drawn:
                obj.new_step(rng)
                self.drawn = True

        def __call__(self, prompts):
            return obj(prompts)

    opt = RandomSwap(ByteTokenizer(), n_candidates=8)
    res = SearchLoop(opt, FixedEpisodes(), steps=10, val_every=100).run(INIT, np.random.default_rng(1))
    losses = [h["loss"] for h in res.history]
    assert all(b <= a + 1e-15 for a, b in zip(losses, losses[1:]))
    assert res.final_prompt != INIT


def test_one_feature_call_per_scoring(population, fake_source):
    obj = _objective(population, fake_source)
    obj.new_step(np.random.default_rng(0))
    n0 = len(fake_source.calls)
    out = obj([b"p1", b"p2", b"p3"])
    assert out.shape == (3,) and len(fake_source.calls) == n0 + 1
    assert fake_source.calls[-1] == (len(obj.rows), 3)


def test_val_best_and_initial_baseline(population, fake_source):
    obj = _objective(population, fake_source)
    train = [i for i, f in enumerate(population.families) if f not in ("fam6", "fam7")]
    va = [i for i, f in enumerate(population.families) if f in ("fam6", "fam7")]
    sigma = standardize_scales(population.S[train])
    val = ValidationObjective(fake_source, [population.specs[i] for i in train], population.S[train],
                              [population.specs[i] for i in va], population.S[va], RidgeReadout(0.1), sigma)
    init_m = val(INIT)
    opt = RandomSwap(ByteTokenizer(), n_candidates=8)
    res = SearchLoop(opt, obj, val=val, steps=9, val_every=3, patience=10).run(INIT, np.random.default_rng(2))
    assert res.best_val_loss <= init_m + 1e-15
    assert np.isclose(val(res.best_prompt), res.best_val_loss)
    assert sum("val_loss" in h for h in res.history) == 3


def test_early_stopping():
    seq = iter([0.5] + [1.0] * 100)
    obj = FuncObjective(_loss_of)
    res = SearchLoop(FixedProposals([b"x" * 30]), obj, val=lambda p: next(seq),
                     steps=100, val_every=2, patience=3).run(b"init" * 10, np.random.default_rng(0))
    assert len(res.history) == 2 * 3  # stops after `patience` validation checks without improvement
    assert res.best_prompt == b"init" * 10 and res.best_val_loss == 0.5
    assert res.init_val_loss == 0.5  # the initial prompt's check is reported separately


def test_init_val_loss_nan_without_val():
    res = SearchLoop(FixedProposals([]), FuncObjective(_loss_of), steps=2).run(b"init" * 10,
                                                                               np.random.default_rng(0))
    assert math.isnan(res.init_val_loss)


def test_early_stopping_resets_on_improvement():
    it = iter([1.0, 0.9, 1.0, 0.8, 1.0, 1.0, 1.0])  # init, then one value per check
    res = SearchLoop(FixedProposals([]), FuncObjective(_loss_of), val=lambda p: next(it),
                     steps=100, val_every=1, patience=2).run(b"a", np.random.default_rng(0))
    assert res.best_val_loss == 0.8


def test_constraints_filter_before_scoring():
    tok = ByteTokenizer()
    c = PromptConstraints(min_bytes=4, max_bytes=10, max_tokens=8, tokenizers=[tok])
    assert c.check(b"abcdef") == (True, "ok")
    assert c.check(b"ab\xff\xfecd")[1] == "invalid_utf8"
    assert c.check(b"abc")[1] == "too_short"
    assert c.check(b"a" * 11)[1] == "too_long"
    assert c.check(b"a" * 9)[1] == "too_many_tokens"  # 9 byte-tokens > 8
    assert c.check("ééééé".encode())[1] == "too_many_tokens"  # 10 bytes
    good, bad_utf8, too_long = b"goodies", b"bad\xff\xfeutf", b"x" * 50
    obj = FuncObjective(_loss_of)
    c2 = PromptConstraints(min_bytes=4, max_bytes=20)
    res = SearchLoop(FixedProposals([bad_utf8, too_long, good, b"ab"]), obj, constraints=c2, steps=3,
                     val_every=100).run(b"initial", np.random.default_rng(0))
    assert bad_utf8 not in obj.evaluated and too_long not in obj.evaluated and b"ab" not in obj.evaluated
    assert good in obj.evaluated
    assert all(h["n_proposed"] == 4 and h["n_valid"] == 1 for h in res.history[:1])


def test_best_of_n_best_is_min_of_all_evaluated():
    obj = FuncObjective(_loss_of)
    opt = BestOfN("random_ascii", length_bytes=40, n_candidates=16)
    res = SearchLoop(opt, obj, steps=10, val_every=100).run(b"i" * 40, np.random.default_rng(0))
    assert len(obj.evaluated) >= 10 * 16
    assert _loss_of(res.best_prompt) == min(_loss_of(p) for p in obj.evaluated)
    assert res.best_step_loss == min(_loss_of(p) for p in obj.evaluated)


@pytest.mark.parametrize("gen", ["random_ascii", "random_bytes", "random_tokens"])
def test_best_of_n_generators_valid(gen):
    opt = BestOfN(gen, length_bytes=64, n_candidates=10, tokenizer=ByteTokenizer() if gen == "random_tokens" else None)
    from onepunch.types import SearchState
    st = SearchState(0, b"", 0.0, b"", 0.0, np.random.default_rng(0))
    ps = opt.propose(st)
    c = PromptConstraints(64, 64)
    assert len(ps) == 10 and all(c.check(p)[0] for p in ps) and len(set(ps)) > 1
    assert opt.propose(SearchState(0, b"", 0.0, b"", 0.0, np.random.default_rng(0))) == ps  # deterministic in rng


def test_random_swap_candidates_valid_and_roundtrip():
    tok = ByteTokenizer()
    opt = RandomSwap(tok, n_candidates=20, allowed="printable_ascii")
    from onepunch.search import roundtrip_ok
    from onepunch.types import SearchState
    ps = opt.propose(SearchState(0, INIT, 0.0, INIT, 0.0, np.random.default_rng(0)))
    assert 0 < len(ps) <= 20
    for p in ps:
        p.decode("utf-8")
        assert roundtrip_ok(tok, tok.encode(p.decode()), p)
        assert p != INIT
    # swaps inside a multibyte char leave invalid UTF-8; those were filtered (decode above succeeded)
    assert 256 not in opt.allowed_ids and len(opt.allowed_ids) == 95


def test_hooks_extras_and_logger(population, fake_source):
    seen = {}

    class Opt:
        def before_step(self, objective, state):
            seen["extras"] = set(state.extras)
            seen["step"] = state.step

        def propose(self, state):
            return [state.prompt + b"!"]

    class StubLogger:
        def __init__(self):
            self.logs, self.tables = [], []

        def log(self, m, step=None):
            self.logs.append((step, m))

        def log_table(self, key, columns, rows, step=None):
            self.tables.append((key, list(columns), list(rows)))

    lg = StubLogger()
    obj = _objective(population, fake_source)
    masks = []

    def sched(t):
        m = population.tokenizers == "tokA"
        masks.append(t)
        return m, "tokA"

    SearchLoop(Opt(), obj, steps=4, val_every=100, logger=lg, log_prompt_every=2,
               active_schedule=sched).run(b"seed prompt", np.random.default_rng(0))
    assert {"episodes", "step_models", "S_step", "sigma"} <= seen["extras"]
    assert masks == [0, 1, 2, 3]
    assert len(lg.logs) == 4 and {"train/loss", "train/loss_before", "search/n_valid"} <= set(lg.logs[0][1])
    assert len(lg.tables) == 2 and len(lg.tables[-1][2]) == 2


@pytest.mark.slow
def test_random_swap_gpt2_roundtrip_filtering():
    import os
    os.environ.setdefault("HF_HOME", "/media/volume/DoubleSteeringLLM/hf_cache")
    from onepunch.search import roundtrip_ok
    from onepunch.search.constraints import allowed_token_ids
    from onepunch.types import SearchState
    opt = RandomSwap("gpt2", n_candidates=30, allowed="all_text")
    tok = opt.tokenizer
    ids = [tok.encode(c)[0] for c in "Hello"]  # H,e,l,l,o as separate tokens: BPE re-merges
    assert not roundtrip_ok(tok, ids, b"Hello")
    assert roundtrip_ok(tok, tok.encode("Hello"), b"Hello")
    allowed = opt.allowed_ids
    assert tok.eos_token_id not in allowed
    assert all("�" not in tok.decode([int(i)]) for i in allowed[::50])
    assert len(allowed_token_ids(tok, "printable_ascii")) < len(allowed)
    p0 = b"The capital of France is Paris, and the cat sat on the mat while it rained."
    ps = opt.propose(SearchState(0, p0, 0.0, p0, 0.0, np.random.default_rng(0)))
    assert len(ps) == 30
    for p in ps:
        p.decode("utf-8")
        assert tok.encode(p.decode()) != tok.encode(p0.decode())
        assert roundtrip_ok(tok, tok.encode(p.decode()), p)
