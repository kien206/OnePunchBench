"""Byte-chunk features (Eq. chunk in formulation_draft.txt). Pure numpy/torch."""

from __future__ import annotations

import numpy as np
import torch

from onepunch.types import TokenLogprobs


def chunk_bounds(n_bytes: int, d: int) -> np.ndarray:
    """Boundaries b[0..d]: chunk j (1-based) covers bytes b[j-1] .. b[j]-1."""
    return (np.arange(d + 1, dtype=np.int64) * int(n_bytes)) // d


def chunk_sizes(n_bytes: int, d: int) -> np.ndarray:
    """|c_j| for j = 1..d; raises if some chunk is empty (n_bytes < d)."""
    sizes = np.diff(chunk_bounds(n_bytes, d))
    if sizes.min() < 1:
        raise ValueError(f"prompt of {n_bytes} bytes is too short for d={d} non-empty chunks")
    return sizes


def chunk_weights(spans: np.ndarray, n_bytes: int, d: int) -> np.ndarray:
    """omega[u, j] = |span_u ∩ c_j| / |span_u|; rows of empty spans are 0."""
    chunk_sizes(n_bytes, d)  # validates
    spans = np.asarray(spans, dtype=np.int64).reshape(-1, 2)
    b = chunk_bounds(n_bytes, d)
    lo = np.maximum(spans[:, :1], b[None, :-1])
    hi = np.minimum(spans[:, 1:], b[None, 1:])
    overlap = np.clip(hi - lo, 0, None).astype(np.float64)
    length = (spans[:, 1] - spans[:, 0]).astype(np.float64)
    return np.divide(overlap, length[:, None], out=np.zeros_like(overlap), where=length[:, None] > 0)


def chunk_features(tl: TokenLogprobs, d: int) -> np.ndarray:
    """f_j = (1/|c_j|) sum_u omega_uj logprob_u, float64 array [d]."""
    w = chunk_weights(tl.spans, tl.n_bytes, d)
    return (w.T @ tl.logprobs.astype(np.float64)) / chunk_sizes(tl.n_bytes, d)


def chunk_features_torch(logprobs: torch.Tensor, weights: torch.Tensor, chunk_sizes: torch.Tensor) -> torch.Tensor:
    """Differentiable version. Shapes [..., n], [..., n, d], [..., d] -> [..., d]."""
    return torch.einsum("...n,...nd->...d", logprobs, weights) / chunk_sizes
