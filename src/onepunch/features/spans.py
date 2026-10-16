"""Byte spans of tokens: which bytes of the prompt does each token cover?

Per-token byte content comes from the vocabulary piece (GPT-2 byte table for
byte-level BPE; ``▁`` -> space and ``<0xNN>`` byte-fallback for SentencePiece).
The concatenation is then aligned to the prompt: it must equal ``p``, optionally
preceded by one tokenizer-added space (SentencePiece dummy prefix). Added bytes
belong to no chunk, so the span of the token holding them is clipped.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

import numpy as np

_BYTE_PIECE = re.compile(r"^<0x([0-9A-Fa-f]{2})>$")
_SP_SPACE = "▁"  # ▁


@lru_cache(maxsize=1)
def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte -> printable unicode char table."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def _decoder_types(node: Any) -> set[str]:
    out: set[str] = set()
    if isinstance(node, dict):
        if "type" in node:
            out.add(node["type"])
        for v in node.values():
            out |= _decoder_types(v)
    elif isinstance(node, list):
        for v in node:
            out |= _decoder_types(v)
    return out


def tokenizer_kind(tokenizer: Any) -> str:
    """'bytelevel' or 'sp' (SentencePiece-style); raises for anything else."""
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        types = _decoder_types(json.loads(backend.to_str()).get("decoder"))
        if "ByteLevel" in types:
            return "bytelevel"
        if types & {"ByteFallback", "Metaspace"}:
            return "sp"
    elif hasattr(tokenizer, "byte_decoder"):
        return "bytelevel"
    elif hasattr(tokenizer, "sp_model"):
        return "sp"
    vocab = tokenizer.get_vocab()
    if "Ġ" in vocab:
        return "bytelevel"
    if any(_SP_SPACE in t for t in vocab):
        return "sp"
    raise NotImplementedError(f"unsupported tokenizer type for byte spans: {type(tokenizer).__name__}")


class SpanTokenizer:
    """Wraps a HF tokenizer: text bytes -> (token ids, byte spans within the text)."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.kind = tokenizer_kind(tokenizer)
        self._decode_table = {c: b for b, c in _bytes_to_unicode().items()}
        self._cache: dict[int, bytes] = {}

    def _piece_bytes(self, token_id: int) -> bytes:
        if token_id in self._cache:
            return self._cache[token_id]
        piece = self.tokenizer.convert_ids_to_tokens(token_id)
        if self.kind == "bytelevel":
            try:
                out = bytes(self._decode_table[c] for c in piece)
            except KeyError:  # special token or non-byte-level piece: literal text
                out = piece.encode("utf-8")
        else:
            m = _BYTE_PIECE.match(piece)
            out = bytes([int(m.group(1), 16)]) if m else piece.replace(_SP_SPACE, " ").encode("utf-8")
        self._cache[token_id] = out
        return out

    def encode(self, text: bytes) -> list[int]:
        """Token ids of `text` (no special tokens added, special-token strings are plain text)."""
        s = text.decode("utf-8")  # strict: invalid UTF-8 cannot be tokenized
        try:
            ids = self.tokenizer(s, add_special_tokens=False, split_special_tokens=True)["input_ids"]
        except TypeError:
            ids = self.tokenizer(s, add_special_tokens=False)["input_ids"]
        return list(ids)

    def encode_with_spans(self, text: bytes) -> tuple[list[int], np.ndarray]:
        """Return (ids, spans [n, 2] int32) with spans in bytes of `text`."""
        ids = self.encode(text)
        pieces = [self._piece_bytes(i) for i in ids]
        joined = b"".join(pieces)
        extra = len(joined) - len(text)
        if extra < 0 or joined[extra:] != text or joined[:extra] not in (b"", b" "):
            raise ValueError(
                "token bytes do not reproduce the text (tokenizer normalizes it?): "
                f"text={text[:60]!r} tokens={joined[:60]!r}"
            )
        ends = np.cumsum([len(b) for b in pieces], dtype=np.int64)
        starts = ends - np.array([len(b) for b in pieces], dtype=np.int64)
        spans = np.stack([starts, ends], axis=1) - extra
        return ids, np.clip(spans, 0, len(text)).astype(np.int32).reshape(-1, 2)
