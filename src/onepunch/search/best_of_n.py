"""Best-of-N baseline: fresh random prompts every step; the loop's keep-best gives best-of-(N*T)."""

from __future__ import annotations

import unicodedata
from typing import Any

import numpy as np

from onepunch.search.constraints import allowed_token_ids, decode, load_tokenizer
from onepunch.types import SearchState

_ASCII = np.arange(0x20, 0x7F, dtype=np.uint8)


def _fit_len(text: str, length_bytes: int, rng: np.random.Generator) -> bytes:
    """NFC-normalize (prompts must be NFC, see PromptConstraints), truncate to <= length_bytes on a
    char boundary, pad with printable ASCII to the exact length."""
    b = unicodedata.normalize("NFC", text).encode("utf-8")[:length_bytes]
    b = b.decode("utf-8", errors="ignore").encode("utf-8")
    return b + rng.choice(_ASCII, size=length_bytes - len(b)).tobytes()


class BestOfN:
    """Optimizer proposing `n_candidates` random prompts of `length_bytes` bytes per step.

    generator: random_ascii (printable ASCII bytes), random_bytes (random bytes made valid UTF-8,
    control chars removed), random_tokens (random allowed tokens of `tokenizer`, concatenated).
    """

    def __init__(self, generator: str = "random_ascii", length_bytes: int = 2048, n_candidates: int = 32,
                 tokenizer: str | Any | None = None):
        if generator not in ("random_ascii", "random_bytes", "random_tokens"):
            raise ValueError(f"unknown generator {generator!r}")
        if generator == "random_tokens" and tokenizer is None:
            raise ValueError("random_tokens needs a tokenizer")
        self.generator, self.length_bytes, self.n_candidates = generator, int(length_bytes), int(n_candidates)
        self._tok_spec, self._tok, self._ids = tokenizer, None, None

    def _one(self, rng: np.random.Generator) -> bytes:
        L = self.length_bytes
        if self.generator == "random_ascii":
            return rng.choice(_ASCII, size=L).tobytes()
        if self.generator == "random_bytes":
            raw = rng.integers(0, 256, size=4 * L, dtype=np.uint8).tobytes()
            text = "".join(c for c in raw.decode("utf-8", errors="ignore") if c.isprintable())
            return _fit_len(text, L, rng)
        if self._tok is None:
            self._tok = load_tokenizer(self._tok_spec)
            self._ids = allowed_token_ids(self._tok, "all_text")
        parts, n = [], 0
        while n < L:
            piece = decode(self._tok, [int(rng.choice(self._ids))])
            parts.append(piece)
            n += len(piece.encode("utf-8"))
        return _fit_len("".join(parts), L, rng)

    def propose(self, state: SearchState) -> list[bytes]:
        return [self._one(state.rng) for _ in range(self.n_candidates)]
