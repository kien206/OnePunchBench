"""Keeps recently used models' weights in pinned host RAM so reloading to GPU is fast."""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch

from onepunch.features.hf_backend import HFBackend, load_model, load_tokenizer
from onepunch.types import ModelSpec

log = logging.getLogger(__name__)
_Key = tuple[str, str, str, str]


def _key(spec: ModelSpec) -> _Key:
    return (spec.model_id, spec.revision, spec.dtype, spec.loader)


@dataclass
class _Entry:
    model: Any
    tokenizer: Any
    tensors: list[torch.Tensor]       # parameters + buffers of the model (their .data is swapped)
    host: list[torch.Tensor]          # the same data on the host (pinned when possible)
    nbytes: int
    cached: bool
    backend: HFBackend | None = field(default=None)


class ModelPool:
    """LRU pool of models whose weights wait in host RAM between uses.

    ``acquire(spec)`` returns a backend with the weights on the GPU; ``release(spec)``
    moves them back to host RAM (weights are frozen, so no copy is needed on the way
    back) or, when ``capacity_gb == 0`` or the model does not fit, frees them (the next
    acquire reloads from disk). Idle models are evicted least-recently-used first
    once the host copies exceed ``capacity_gb``.
    """

    def __init__(self, capacity_gb: float = 0.0, device: str = "cuda", backend_cls: Any = None,
                 pin_memory: bool = False, **backend_kwargs: Any):
        """pin_memory: keep host copies in pinned memory (faster H2D copies). Off by default: torch's
        caching pinned allocator does not return freed blocks to the OS, so evicting models of varying
        sizes grows host RAM well past `capacity_gb` (a 115 GB host was OOM-killed this way)."""
        self.backend_cls = backend_cls or HFBackend
        self.pin_memory = pin_memory
        self.capacity = int(capacity_gb * 2**30)
        self.device = device
        self.backend_kwargs = backend_kwargs
        self._entries: OrderedDict[_Key, _Entry] = OrderedDict()

    # -- helpers -----------------------------------------------------------
    @property
    def host_bytes(self) -> int:
        return sum(e.nbytes for e in self._entries.values() if e.cached)

    def _evict(self) -> None:
        for k in list(self._entries):
            if self.host_bytes <= self.capacity:
                return
            e = self._entries[k]
            if e.backend is None and e.cached:  # idle
                del self._entries[k]
                log.info("pool: evicted %s", k[0])

    def _to_device(self, e: _Entry) -> None:
        for t, h in zip(e.tensors, e.host):
            t.data = h.to(self.device, non_blocking=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # -- API ---------------------------------------------------------------
    def acquire(self, spec: ModelSpec) -> HFBackend:
        k = _key(spec)
        e = self._entries.get(k)
        if e is not None and e.backend is not None:
            return e.backend
        if e is None:
            tok = load_tokenizer(spec)
            if self.capacity == 0:
                model = load_model(spec, self.device, self.backend_kwargs.get("attn_implementation", "sdpa"))
                e = _Entry(model, tok, [], [], 0, cached=False)
            else:
                model = load_model(spec, "cpu", self.backend_kwargs.get("attn_implementation", "sdpa"))
                tensors = list(model.parameters()) + list(model.buffers())
                nbytes = sum(t.numel() * t.element_size() for t in tensors)
                if nbytes > self.capacity:
                    model = model.to(self.device)
                    e = _Entry(model, tok, [], [], nbytes, cached=False)
                else:
                    host = []
                    for t in tensors:
                        if self.pin_memory:
                            try:
                                t.data = t.data.pin_memory()
                            except RuntimeError:  # cannot pin: keep pageable memory
                                pass
                        host.append(t.data)
                    e = _Entry(model, tok, tensors, host, nbytes, cached=True)
                    self._to_device(e)
            self._entries[k] = e
        elif e.cached:
            self._to_device(e)
        self._entries.move_to_end(k)
        e.backend = self.backend_cls(spec, self.device, model=e.model, tokenizer=e.tokenizer, **self.backend_kwargs)
        self._evict()
        return e.backend

    def release(self, spec: ModelSpec) -> None:
        k = _key(spec)
        e = self._entries.get(k)
        if e is None:
            return
        if e.backend is not None:
            e.backend.close()
            e.backend = None
        if e.cached:
            for t, h in zip(e.tensors, e.host):
                t.data = h
        else:
            del self._entries[k]
        del e
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._evict()

    def clear(self) -> None:
        """Release everything, including host copies."""
        for k in list(self._entries):
            self.release(ModelSpec(model_id=k[0], revision=k[1], dtype=k[2], loader=k[3]))
        self._entries.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
