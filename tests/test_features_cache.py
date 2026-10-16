import zlib

import numpy as np
import pytest

from onepunch.features.cache import LogprobCache
from onepunch.features.chunking import chunk_features
from onepunch.features.extractor import FeatureExtractor
from onepunch.types import LogprobBackend, ModelSpec, TokenLogprobs

A = ModelSpec("org/model-a", revision="abc", dtype="bfloat16")
B = ModelSpec("org/model-b")


def fake_tl(p: bytes, salt: int = 0) -> TokenLogprobs:
    """One token per 3 bytes, log-probs a deterministic function of the content."""
    rng = np.random.default_rng(zlib.crc32(p) + salt)
    n = len(p)
    starts = np.arange(0, n, 3)
    spans = np.stack([starts, np.minimum(starts + 3, n)], axis=1)
    return TokenLogprobs(-rng.random(len(spans)) * 4, spans, n, f"tok{salt}")


class FakeBackend:
    def __init__(self, spec, log):
        self.spec, self.log, self.closed = spec, log, False

    def token_logprobs(self, prompts, prefix):
        self.log.append((self.spec.model_id, list(prompts), prefix))
        return [fake_tl(p, salt=len(self.spec.model_id)) for p in prompts]

    @property
    def tokenizer(self):
        return None

    def close(self):
        self.closed = True


def make_extractor(tmp_path, d=4, **kw):
    log: list = []
    cache = LogprobCache(tmp_path, persist=kw.pop("persist", True))
    ex = FeatureExtractor(d, cache, backend_factory=lambda s: FakeBackend(s, log), **kw)
    return ex, cache, log


def test_fake_backend_satisfies_protocol():
    assert isinstance(FakeBackend(A, []), LogprobBackend)


def test_roundtrip_persistent(tmp_path):
    cache = LogprobCache(tmp_path)
    p = b"hello world, some prompt bytes"
    tl = fake_tl(p)
    assert cache.get(A, p, b"\n") is None and not cache.has(A, p, b"\n")
    cache.put(A, p, b"\n", tl)
    path = cache.path(A, p, b"\n")
    assert path.exists() and path.parent.parent.name == "logprobs"
    for c in (cache, LogprobCache(tmp_path)):  # memory hit, then a fresh process-like instance
        got = c.get(A, p, b"\n")
        np.testing.assert_array_equal(got.logprobs, tl.logprobs)
        np.testing.assert_array_equal(got.spans, tl.spans)
        assert got.n_bytes == tl.n_bytes and got.tokenizer_hash == tl.tokenizer_hash
    with np.load(path) as z:
        assert str(z["backend"]) == "hf" and str(z["dtype"]) == "bfloat16" and "transformers_version" in z
    # key separates model, revision, dtype, prefix
    assert not cache.has(B, p, b"\n") and not cache.has(A, p, b"\n\n")
    assert not cache.has(ModelSpec("org/model-a", revision="abc", dtype="float32"), p, b"\n")
    assert not cache.has(ModelSpec("org/model-a", revision="other", dtype="bfloat16"), p, b"\n")


def test_persist_false_stays_in_memory(tmp_path):
    cache = LogprobCache(tmp_path, persist=False)
    p = b"candidate prompt"
    cache.put(A, p, b"\n", fake_tl(p))
    assert cache.has(A, p, b"\n") and not list(tmp_path.rglob("*.npz"))
    assert LogprobCache(tmp_path).get(A, p, b"\n") is None
    cache.put(A, p, b"\n", fake_tl(p), persist=True)  # per-call override
    assert len(list(tmp_path.rglob("*.npz"))) == 1


def test_memory_lru_evicts_oldest(tmp_path):
    cache = LogprobCache(tmp_path, persist=False, max_memory_items=2)
    ps = [b"p1", b"p2", b"p3"]
    for p in ps:
        cache.put(A, p, b"\n", fake_tl(p))
    assert cache.missing(A, ps, b"\n") == [0]


def test_extractor_computes_only_missing_pairs(tmp_path):
    ex, cache, log = make_extractor(tmp_path)
    ps = [bytes([65 + i]) * 40 for i in range(5)]
    phi = ex.features([A, B], ps[:3])
    assert phi.shape == (2, 3, 4) and phi.dtype == np.float64
    assert [(m, len(p)) for m, p, _ in log] == [("org/model-a", 3), ("org/model-b", 3)]  # model-major, one load each
    assert log[0][2] == b"\n"

    log.clear()
    ex.features([A, B], ps[:3])
    assert log == []  # fully cached

    phi2 = ex.features([A, B], ps)  # two new prompts, for both models
    assert [(m, p) for m, p, _ in log] == [("org/model-a", ps[3:]), ("org/model-b", ps[3:])]
    np.testing.assert_array_equal(phi2[:, :3], phi)

    log.clear()
    ex2 = FeatureExtractor(4, LogprobCache(tmp_path), backend_factory=lambda s: FakeBackend(s, log))
    np.testing.assert_allclose(ex2.features([B, A], ps)[1], phi2[0])  # persisted across instances
    assert log == []


def test_extractor_dedups_and_matches_chunk_features(tmp_path):
    ex, cache, log = make_extractor(tmp_path, d=8)
    p = b"x" * 64
    q = b"y" * 70
    tls = ex.token_logprobs([A], [p, q, p])
    assert [len(x) for _, x, _ in log] == [2]  # p computed once
    assert tls[0][0] is tls[0][2] or np.array_equal(tls[0][0].logprobs, tls[0][2].logprobs)
    phi = ex.features([A], [p, q, p])
    np.testing.assert_allclose(phi[0, 0], chunk_features(tls[0][0], 8))
    np.testing.assert_array_equal(phi[0, 0], phi[0, 2])


def test_extractor_candidates_not_persisted_and_backend_closed(tmp_path):
    made: list = []
    cache = LogprobCache(tmp_path)

    def factory(spec):
        made.append(FakeBackend(spec, []))
        return made[-1]

    ex = FeatureExtractor(2, cache, backend_factory=factory, persist=False)
    ex.features([A], [b"candidate bytes"])
    assert not list(tmp_path.rglob("*.npz")) and made[0].closed and cache.has(A, b"candidate bytes", b"\n")


def test_min_chunk_length_enforced(tmp_path):
    ex, _, log = make_extractor(tmp_path, d=4, l_chunk=32)
    with pytest.raises(ValueError):
        ex.features([A], [b"z" * 127])
    assert log == []
    ex.features([A], [b"z" * 128])


def test_dtype_override_changes_cache_key(tmp_path):
    ex, cache, log = make_extractor(tmp_path, dtype="float32")
    ex.features([A], [b"w" * 20])
    assert cache.has(ModelSpec("org/model-a", "abc", "float32"), b"w" * 20, b"\n")
    assert not cache.has(A, b"w" * 20, b"\n")
