"""Random single-token swap proposals (the gradient-free control for GCG-style search)."""

from __future__ import annotations

from typing import Any

import numpy as np

from onepunch.search.constraints import allowed_token_ids, decode, encode, load_tokenizer, roundtrip_ok
from onepunch.types import SearchState


class RandomSwap:
    """Optimizer: tokenize p, replace one random position by a random allowed token, detokenize,
    keep only candidates that round-trip (tok(p~) == ids) and are valid UTF-8."""

    def __init__(self, tokenizer: str | Any, n_candidates: int = 32, allowed: str = "printable_ascii",
                 max_tries: int = 10):
        self._tok_spec = tokenizer
        self._tok: Any = None
        self._allowed_ids: np.ndarray | None = None
        self.n_candidates, self.allowed, self.max_tries = int(n_candidates), allowed, int(max_tries)

    @property
    def tokenizer(self) -> Any:
        if self._tok is None:
            self._tok = load_tokenizer(self._tok_spec)
        return self._tok

    @property
    def allowed_ids(self) -> np.ndarray:
        if self._allowed_ids is None:
            self._allowed_ids = allowed_token_ids(self.tokenizer, self.allowed)
        return self._allowed_ids

    def propose(self, state: SearchState) -> list[bytes]:
        tok, rng = self.tokenizer, state.rng
        ids = encode(tok, state.prompt.decode("utf-8"))
        if not ids:
            return []
        out: list[bytes] = []
        seen = {state.prompt}
        for _ in range(self.n_candidates * self.max_tries):
            if len(out) >= self.n_candidates:
                break
            new = list(ids)
            new[int(rng.integers(len(ids)))] = int(rng.choice(self.allowed_ids))
            try:
                p = decode(tok, new).encode("utf-8")
            except UnicodeEncodeError:
                continue
            if p in seen or not roundtrip_ok(tok, new, p):
                continue
            seen.add(p)
            out.append(p)
        return out
