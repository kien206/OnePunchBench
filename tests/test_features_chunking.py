import numpy as np
import pytest
import torch

from onepunch.features.chunking import (
    chunk_bounds, chunk_features, chunk_features_torch, chunk_sizes, chunk_weights,
)
from onepunch.types import TokenLogprobs


def tiling_spans(n_bytes, lengths):
    ends = np.cumsum(lengths)
    assert ends[-1] == n_bytes
    return np.stack([ends - np.array(lengths), ends], axis=1)


def test_bounds_follow_floor_formula():
    for n, d in [(100, 16), (2048, 64), (33, 1), (17, 16), (1000, 7)]:
        b = chunk_bounds(n, d)
        assert b[0] == 0 and b[-1] == n and len(b) == d + 1
        for j in range(1, d + 1):
            assert b[j - 1] == (j - 1) * n // d and b[j] == j * n // d
        assert chunk_sizes(n, d).sum() == n


def test_empty_chunk_raises():
    with pytest.raises(ValueError):
        chunk_sizes(5, 16)
    with pytest.raises(ValueError):
        chunk_weights(np.array([[0, 5]]), 5, 16)


@pytest.mark.parametrize("d", [1, 4, 16, 64])
def test_sum_identity_when_spans_tile(d):
    rng = np.random.default_rng(0)
    lengths = rng.integers(1, 9, size=300)
    n = int(lengths.sum())
    tl = TokenLogprobs(-rng.random(len(lengths)) * 5, tiling_spans(n, lengths), n, "h")
    f = chunk_features(tl, d)
    assert f.shape == (d,)
    np.testing.assert_allclose((f * chunk_sizes(n, d)).sum(), tl.logprobs.astype(np.float64).sum(), rtol=1e-9)


def test_straddling_token_splits_proportionally():
    # n=8, d=2 -> chunks [0,4) [4,8); token 1 covers [2,8): 2 bytes in c1, 4 in c2
    spans = np.array([[0, 2], [2, 8]])
    w = chunk_weights(spans, 8, 2)
    np.testing.assert_allclose(w, [[1, 0], [1 / 3, 2 / 3]])
    tl = TokenLogprobs([-1.0, -3.0], spans, 8, "h")
    np.testing.assert_allclose(chunk_features(tl, 2), [(-1 - 1) / 4, (-2) / 4])


def test_zero_length_span_ignored_and_dummy_prefix_excluded():
    # dummy-prefix token has an empty span; its logprob belongs to no chunk
    spans = np.array([[0, 0], [0, 4], [4, 8]])
    w = chunk_weights(spans, 8, 2)
    assert (w[0] == 0).all()
    tl = TokenLogprobs([-100.0, -1.0, -2.0], spans, 8, "h")
    np.testing.assert_allclose(chunk_features(tl, 2), [-0.25, -0.5])


def test_torch_matches_numpy_and_is_differentiable():
    rng = np.random.default_rng(1)
    lengths = rng.integers(1, 6, size=80)
    n = int(lengths.sum())
    spans = tiling_spans(n, lengths)
    lp = -rng.random(len(lengths))
    d = 8
    w = torch.tensor(chunk_weights(spans, n, d))
    sizes = torch.tensor(chunk_sizes(n, d), dtype=torch.float64)
    x = torch.tensor(lp, requires_grad=True)
    f = chunk_features_torch(x, w, sizes)
    np.testing.assert_allclose(f.detach().numpy(), chunk_features(TokenLogprobs(lp, spans, n, "h"), d), rtol=1e-5)
    f.sum().backward()
    np.testing.assert_allclose(x.grad.numpy(), (w / sizes).sum(1).numpy())
    # batched
    fb = chunk_features_torch(torch.stack([x, x]), torch.stack([w, w]), torch.stack([sizes, sizes]))
    assert fb.shape == (2, d)
