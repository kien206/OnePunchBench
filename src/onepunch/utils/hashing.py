"""Stable hashes used as cache keys and identity checks across subpackages."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def sha256_bytes(data: bytes, n: int = 16) -> str:
    return hashlib.sha256(data).hexdigest()[:n]


def prompt_hash(prompt: bytes) -> str:
    """Cache key for a prompt (or prefix)."""
    return sha256_bytes(prompt)


def config_hash(obj: Any) -> str:
    """Hash of a JSON-serializable config (dict keys sorted)."""
    return sha256_bytes(json.dumps(obj, sort_keys=True, default=str).encode())


def tokenizer_hash(tokenizer: Any) -> str:
    """Identity of a tokenizer *as it splits plain text*.

    Two tokenizers get the same hash iff their normalizer, pre-tokenizer,
    model (vocab + merges) and decoder agree. Added/special tokens and the
    post-processor are ignored, so e.g. Qwen2.5 and Qwen3 (same BPE, different
    chat tokens) share a hash. Falls back to the sorted vocab for slow
    tokenizers.
    """
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        spec = json.loads(backend.to_str())
        keep = {k: spec.get(k) for k in ("normalizer", "pre_tokenizer", "model", "decoder")}
        return config_hash(keep)
    vocab = tokenizer.get_vocab()
    return config_hash({"cls": type(tokenizer).__name__, "vocab": sorted(vocab.items())})
