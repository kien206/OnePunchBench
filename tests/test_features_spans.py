"""Byte spans for byte-level BPE and SentencePiece tokenizers (tokenizer downloads only)."""

import numpy as np
import pytest

pytestmark = pytest.mark.slow

TOKENIZERS = {
    "HuggingFaceTB/SmolLM2-135M": "bytelevel",
    "Qwen/Qwen2.5-0.5B": "bytelevel",
    "EleutherAI/pythia-70m": "bytelevel",
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T": "sp",  # Llama-2 tokenizer: dummy prefix, byte fallback
    "unsloth/gemma-2-2b": "sp",                                    # Gemma tokenizer: no dummy prefix, byte fallback
}

PROMPTS = [
    "Hello world, this is plain ASCII text.",
    " leading space and trailing space ",
    "line one\n\nline three\n\n\nline six\n",
    "tab\tseparated\tvalues\r\nCRLF line",
    "日本語のテキストと漢字、中文也可以。",
    "emoji 😀👍🏽🇻🇳 and a ZWJ family 👨‍👩‍👧 end",
    "Tiếng Việt: Nguyễn Thị Hồng đang học tiếng Anh, ạ ế ỗ ữ",
    "mixed é à ü ß ñ Ж Ω العربية",
    "literal specials <|endoftext|> <s> </s> <unk> [INST] here",
]


@pytest.fixture(scope="module", params=list(TOKENIZERS))
def span_tok(request):
    from transformers import AutoTokenizer

    from onepunch.features.spans import SpanTokenizer

    st = SpanTokenizer(AutoTokenizer.from_pretrained(request.param))
    assert st.kind == TOKENIZERS[request.param]
    return st


@pytest.mark.parametrize("text", PROMPTS)
def test_spans_tile_prompt_and_recover_bytes(span_tok, text):
    p = text.encode("utf-8")
    ids, spans = span_tok.encode_with_spans(p)
    assert len(ids) == len(spans) > 0
    # ordered, non-overlapping, covering [0, n) exactly
    assert (spans[:, 0] <= spans[:, 1]).all()
    assert spans[0, 0] == 0 and spans[-1, 1] == len(p)
    assert (spans[1:, 0] == spans[:-1, 1]).all()
    assert (np.diff(spans, axis=1) >= 0).all()
    # the span slices of p reproduce p, and equal the token bytes (up to the tokenizer-added prefix)
    assert b"".join(p[s:e] for s, e in spans) == p
    pieces = [span_tok._piece_bytes(i) for i in ids]
    extra = sum(map(len, pieces)) - len(p)
    assert extra in (0, 1)
    assert b"".join(pieces)[extra:] == p
    # at most one token carries the tokenizer-added byte, and only the first one
    short = [k for k, (s, e) in enumerate(spans) if e - s < len(pieces[k])]
    assert short == ([0] if extra else [])


def test_special_token_strings_are_plain_text(span_tok):
    ids = span_tok.encode(b"<|endoftext|> <s> </s>")
    assert not set(ids) & set(span_tok.tokenizer.all_special_ids)


def test_multibyte_split_across_tokens(span_tok):
    # emoji / CJK are usually split into several tokens, each with a partial-character span
    p = "😀日本".encode("utf-8")
    ids, spans = span_tok.encode_with_spans(p)
    assert b"".join(p[s:e] for s, e in spans) == p
    assert len(p) == 10


def test_dummy_prefix_excluded_for_sentencepiece():
    from transformers import AutoTokenizer

    from onepunch.features.spans import SpanTokenizer

    st = SpanTokenizer(AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"))
    p = b"Hello world"
    ids, spans = st.encode_with_spans(p)
    assert sum(len(st._piece_bytes(i)) for i in ids) == len(p) + 1  # "▁Hello" carries the dummy space
    assert spans[0, 0] == 0 and spans[0, 1] == 5                      # ... but its span is just "Hello"
