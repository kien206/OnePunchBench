"""Log-prob cache: in-memory LRU on top of persistent per-(model, prompt) npz files."""

from __future__ import annotations

import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np
import transformers

from onepunch.types import ModelSpec, TokenLogprobs
from onepunch.utils.hashing import prompt_hash

Key = tuple[str, str, str, str, str]  # model_id, revision, dtype, hash(p), hash(p0)


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "__", s)


class LogprobCache:
    """Cache of TokenLogprobs keyed by (model_id, revision, dtype, hash(p), hash(p0)).

    Disk layout: ``<root>/logprobs/<model_slug>@<revision>.<dtype>/<p_hash>_<p0_hash>.npz``.
    With ``persist=False`` (e.g. search candidates) entries live in memory only.
    ``put(..., persist=...)`` overrides the default per call.
    """

    def __init__(self, root: str | os.PathLike, persist: bool = True, max_memory_items: int = 200_000,
                 backend: str = "hf"):
        self.root = Path(root)
        self.persist = persist
        self.max_memory_items = max_memory_items
        self.backend = backend
        self._mem: OrderedDict[Key, TokenLogprobs] = OrderedDict()

    @staticmethod
    def key(spec: ModelSpec, prompt: bytes, prefix: bytes) -> Key:
        return (spec.model_id, spec.revision, spec.dtype, prompt_hash(prompt), prompt_hash(prefix))

    def path(self, spec: ModelSpec, prompt: bytes, prefix: bytes) -> Path:
        _, rev, dtype, ph, p0h = self.key(spec, prompt, prefix)
        return self.root / "logprobs" / f"{_slug(spec.model_id)}@{_slug(rev)}.{dtype}" / f"{ph}_{p0h}.npz"

    def _remember(self, key: Key, tl: TokenLogprobs) -> None:
        self._mem[key] = tl
        self._mem.move_to_end(key)
        while len(self._mem) > self.max_memory_items:
            self._mem.popitem(last=False)

    def get(self, spec: ModelSpec, prompt: bytes, prefix: bytes) -> TokenLogprobs | None:
        key = self.key(spec, prompt, prefix)
        if key in self._mem:
            self._mem.move_to_end(key)
            return self._mem[key]
        path = self.path(spec, prompt, prefix)
        if not path.exists():
            return None
        try:
            with np.load(path, allow_pickle=False) as z:
                tl = TokenLogprobs(z["logprobs"], z["spans"], int(z["n_bytes"]), str(z["tokenizer_hash"]))
        except Exception:  # truncated / corrupt file: treat as a miss, it will be recomputed
            return None
        self._remember(key, tl)
        return tl

    def has(self, spec: ModelSpec, prompt: bytes, prefix: bytes) -> bool:
        return self.key(spec, prompt, prefix) in self._mem or self.path(spec, prompt, prefix).exists()

    def put(self, spec: ModelSpec, prompt: bytes, prefix: bytes, tl: TokenLogprobs,
            persist: bool | None = None) -> None:
        self._remember(self.key(spec, prompt, prefix), tl)
        if self.persist if persist is None else persist:
            path = self.path(spec, prompt, prefix)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.stem}.{os.getpid()}.tmp.npz")
            np.savez(tmp, logprobs=tl.logprobs, spans=tl.spans, n_bytes=tl.n_bytes,
                     tokenizer_hash=tl.tokenizer_hash, backend=self.backend, dtype=spec.dtype,
                     transformers_version=transformers.__version__)
            os.replace(tmp, path)  # atomic: readers never see partial files

    def missing(self, spec: ModelSpec, prompts: Sequence[bytes], prefix: bytes) -> list[int]:
        """Indices of prompts that are not cached for `spec`."""
        return [i for i, p in enumerate(prompts) if not self.has(spec, p, prefix)]
