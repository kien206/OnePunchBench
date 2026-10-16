"""FeatureExtractor: model-major, cached computation of chunk features (a FeatureSource)."""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Callable, Sequence

import numpy as np

from onepunch.features.cache import LogprobCache
from onepunch.features.chunking import chunk_features
from onepunch.features.model_pool import ModelPool
from onepunch.types import LogprobBackend, ModelSpec, TokenLogprobs

log = logging.getLogger(__name__)

_NOT_BACKEND_KEYS = {"_target_", "_partial_", "pool_ram_gb", "dtype"}


class FeatureExtractor:
    """Implements ``onepunch.types.FeatureSource``.

    Exactly one of `pool` (a ModelPool) and `backend_factory` (spec -> LogprobBackend,
    closed after use) supplies backends. For each model, all of its uncached prompts
    are scored in one load, written to the cache, and the model is released.

    `l_chunk` is the minimum chunk length: prompts shorter than ``d * l_chunk``
    bytes are rejected. `persist=False` keeps results in memory only (search
    candidates). `dtype` overrides the spec's dtype (part of the cache key).
    """

    def __init__(
        self,
        d: int,
        cache: LogprobCache,
        pool: ModelPool | None = None,
        backend_factory: Callable[[ModelSpec], LogprobBackend] | None = None,
        prefix: bytes = b"\n",
        l_chunk: int = 0,
        persist: bool | None = None,
        dtype: str | None = None,
    ):
        if (pool is None) == (backend_factory is None):
            raise ValueError("give exactly one of pool / backend_factory")
        self.d, self.cache, self.pool, self.backend_factory = d, cache, pool, backend_factory
        self.prefix, self.l_chunk, self.persist, self.dtype = prefix, l_chunk, persist, dtype

    @classmethod
    def from_cfg(cls, cfg: Any, persist: bool | None = None) -> "FeatureExtractor":
        """Build from the Hydra config: `cfg.backend`, `cfg.features`, `cfg.paths.cache`."""
        from hydra.utils import get_class

        b, f = cfg.backend, cfg.features
        kwargs = {k: v for k, v in dict(b).items() if k not in _NOT_BACKEND_KEYS and k != "device"}
        backend_cls = get_class(b._target_) if "_target_" in b else None
        pool = ModelPool(b.get("pool_ram_gb", 0) or 0, b.get("device", "cuda"), backend_cls=backend_cls, **kwargs)
        cache = LogprobCache(cfg.paths.cache, persist=bool(f.get("persist", True)))
        prefix = str(f.get("prefix", "\n")).encode()
        return cls(f.d, cache, pool=pool, prefix=prefix, l_chunk=int(f.get("l_chunk", 0)),
                   persist=persist, dtype=b.get("dtype"))

    # -- internals ---------------------------------------------------------
    def _spec(self, m: ModelSpec) -> ModelSpec:
        return dataclasses.replace(m, dtype=self.dtype) if self.dtype else m

    def _check(self, prompts: Sequence[bytes]) -> None:
        need = max(self.d, self.d * self.l_chunk)
        for p in prompts:
            if len(p) < need:
                raise ValueError(f"prompt of {len(p)} bytes is shorter than d*l_chunk = {need}")

    def _compute(self, spec: ModelSpec, prompts: list[bytes]) -> None:
        t0 = time.time()
        backend = self.pool.acquire(spec) if self.pool is not None else self.backend_factory(spec)  # type: ignore[misc]
        try:
            results = backend.token_logprobs(prompts, self.prefix)
        finally:
            if self.pool is not None:
                self.pool.release(spec)
            else:
                backend.close()
        for p, tl in zip(prompts, results):
            self.cache.put(spec, p, self.prefix, tl, persist=self.persist)
        log.info("scored %d prompts with %s in %.1fs", len(prompts), spec.model_id, time.time() - t0)

    # -- FeatureSource -----------------------------------------------------
    def token_logprobs(self, models: Sequence[ModelSpec], prompts: Sequence[bytes]) -> list[list[TokenLogprobs]]:
        """[model][prompt] -> TokenLogprobs; computes only missing pairs, model-major."""
        prompts = list(prompts)
        self._check(prompts)
        unique = list(dict.fromkeys(prompts))
        out = []
        for i, m in enumerate(models):
            spec = self._spec(m)
            todo = [unique[j] for j in self.cache.missing(spec, unique, self.prefix)]
            log.info("[%d/%d] %s: %d/%d prompts to compute", i + 1, len(models), spec.model_id, len(todo), len(unique))
            if todo:
                self._compute(spec, todo)
            out.append([self.cache.get(spec, p, self.prefix) for p in prompts])
        return out  # type: ignore[return-value]

    def features(self, models: Sequence[ModelSpec], prompts: Sequence[bytes]) -> np.ndarray:
        """Phi [N_models, N_prompts, d] float64."""
        tls = self.token_logprobs(models, prompts)
        phi = np.empty((len(models), len(prompts), self.d), dtype=np.float64)
        for i, row in enumerate(tls):
            for j, tl in enumerate(row):
                phi[i, j] = chunk_features(tl, self.d)
        return phi
