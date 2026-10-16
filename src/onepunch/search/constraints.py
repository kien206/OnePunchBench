"""Feasible prompt set P and tokenizer helpers (formulation: Bilevel problem / Search)."""

from __future__ import annotations

import unicodedata
from typing import Any, Sequence

import numpy as np


def load_tokenizer(tokenizer: Any) -> Any:
    """A tokenizer object, or a HF name loaded lazily with AutoTokenizer."""
    if isinstance(tokenizer, str):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tokenizer)
    return tokenizer


def encode(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=False))


def decode(tokenizer: Any, ids: Sequence[int]) -> str:
    try:
        return tokenizer.decode(list(ids), clean_up_tokenization_spaces=False)
    except TypeError:
        return tokenizer.decode(list(ids))


def roundtrip_ok(tokenizer: Any, ids: Sequence[int], p: bytes) -> bool:
    """tok(p) == ids: catches BPE re-merging around a swap. False if p is not valid UTF-8."""
    try:
        text = p.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return encode(tokenizer, text) == list(ids)


def allowed_token_ids(tokenizer: Any, mode: str = "printable_ascii") -> np.ndarray:
    """Token ids that are not special and decode to complete UTF-8 text.

    mode "printable_ascii" additionally keeps only tokens made of chars 0x20-0x7e.
    """
    if mode not in ("printable_ascii", "all_text"):
        raise ValueError(f"unknown allowed mode {mode!r}")
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    if hasattr(tokenizer, "get_added_vocab"):
        special |= set(tokenizer.get_added_vocab().values())
    ids = [i for i in range(len(tokenizer)) if i not in special]
    if hasattr(tokenizer, "batch_decode"):
        try:
            texts = tokenizer.batch_decode([[i] for i in ids], clean_up_tokenization_spaces=False)
        except TypeError:
            texts = tokenizer.batch_decode([[i] for i in ids])
    else:
        texts = [decode(tokenizer, [i]) for i in ids]
    keep = []
    for i, s in zip(ids, texts):
        if not s or "�" in s or any(ord(c) < 0x20 or ord(c) == 0x7f for c in s):
            continue
        if mode == "printable_ascii" and not all(0x20 <= ord(c) <= 0x7e for c in s):
            continue
        keep.append(i)
    return np.array(keep, dtype=np.int64)


class PromptConstraints:
    """Feasible set: valid UTF-8, min_bytes <= |p| <= max_bytes, <= max_tokens under every tokenizer.

    min_bytes is d * L_chunk (computed by the caller). `tokenizers` are tokenizer objects or HF names.
    """

    def __init__(self, min_bytes: int, max_bytes: int, max_tokens: int | None = None,
                 tokenizers: Sequence[Any] = ()):
        self.min_bytes, self.max_bytes, self.max_tokens = int(min_bytes), int(max_bytes), max_tokens
        self.tokenizers = [load_tokenizer(t) for t in tokenizers] if max_tokens else []

    def check(self, p: bytes) -> tuple[bool, str]:
        try:
            text = p.decode("utf-8")
        except UnicodeDecodeError:
            return False, "invalid_utf8"
        if unicodedata.normalize("NFC", text) != text:  # NFC-normalizing tokenizers (Qwen, Pythia) would read other bytes
            return False, "not_nfc"
        if len(p) < self.min_bytes:
            return False, "too_short"
        if len(p) > self.max_bytes:
            return False, "too_long"
        if self.max_tokens is not None:
            for t in self.tokenizers:
                if len(encode(t, text)) > self.max_tokens:
                    return False, "too_many_tokens"
        return True, "ok"

    def filter(self, prompts: Sequence[bytes]) -> list[bytes]:
        return [p for p in prompts if self.check(p)[0]]
