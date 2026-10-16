"""HuggingFace transformers backend: per-token log-probs of prompts under one model."""

from __future__ import annotations

import contextlib
import gc
import logging
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from onepunch.features.spans import SpanTokenizer
from onepunch.types import ModelSpec, TokenLogprobs
from onepunch.utils import hashing

log = logging.getLogger(__name__)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def torch_dtype(name: str) -> torch.dtype:
    return _DTYPES[name]


def load_model(spec: ModelSpec, device: str = "cpu", attn_implementation: str = "sdpa") -> Any:
    """Load the causal LM of `spec` (eval mode, frozen) on `device`."""
    if spec.loader == "hf_vlm_text":
        raise NotImplementedError("loader 'hf_vlm_text' (text tower of a VLM) is not implemented yet")
    if spec.loader != "hf":
        raise NotImplementedError(f"unknown loader {spec.loader!r}")
    kwargs = dict(revision=spec.revision, dtype=torch_dtype(spec.dtype), trust_remote_code=spec.trust_remote_code)
    try:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            spec.model_id, attn_implementation=attn_implementation, **kwargs)
    except ValueError as e:  # e.g. GPT-Neo has no SDPA kernel
        if attn_implementation == "eager" or "attention implementation" not in str(e):
            raise
        log.warning("%s: %s unsupported, falling back to eager attention", spec.model_id, attn_implementation)
        model = transformers.AutoModelForCausalLM.from_pretrained(spec.model_id, attn_implementation="eager", **kwargs)
    model.eval().requires_grad_(False)
    return model.to(device)


def load_tokenizer(spec: ModelSpec) -> Any:
    return transformers.AutoTokenizer.from_pretrained(
        spec.model_id, revision=spec.revision, trust_remote_code=spec.trust_remote_code
    )


@contextlib.contextmanager
def fp32_lm_head(model: Any, enabled: bool = True):
    """Compute the output projection in float32 (bf16 logits of size ~20 are rounded to ~0.1).

    Implemented as a forward hook on the LM head, so model-specific post-processing
    (e.g. logit soft-capping) still runs on the float32 logits.
    """
    head = model.get_output_embeddings()
    if not enabled or not isinstance(head, torch.nn.Linear) or head.weight.dtype == torch.float32:
        yield
        return

    def hook(mod: torch.nn.Linear, args: tuple, out: torch.Tensor) -> torch.Tensor:
        b = None if mod.bias is None else mod.bias.float()
        return F.linear(args[0].float(), mod.weight.float(), b)

    handle = head.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def sequence_logprobs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    inputs_embeds: torch.Tensor | None = None,
    chunk_elems: int = 2**28,
    fp32_head: bool = True,
) -> torch.Tensor:
    """out[b, t-1] = log P(input_ids[b, t] | input_ids[b, :t]) for t = 1..T-1, float32.

    The forward uses `inputs_embeds` instead of `input_ids` when given (gradient
    path); `input_ids` still supplies the targets. Log-softmax runs in float32 in
    time chunks, so no [B, T, V] float32 tensor is materialized. Entries at padded
    positions are finite but meaningless; mask them out.
    """
    kw: dict[str, Any] = {"attention_mask": attention_mask}
    if inputs_embeds is not None:
        kw["inputs_embeds"] = inputs_embeds
    else:
        kw["input_ids"] = input_ids
    with fp32_lm_head(model, fp32_head):
        logits = model(**kw).logits[:, :-1]  # [B, T-1, V]; float32 if fp32_head else model dtype
    targets = input_ids[:, 1:]
    B, T, V = logits.shape
    step = max(1, chunk_elems // max(1, B * V))
    outs = []
    for a in range(0, T, step):
        lg = logits[:, a : a + step].float()
        tgt = targets[:, a : a + step].unsqueeze(-1)
        outs.append(lg.gather(-1, tgt).squeeze(-1) - torch.logsumexp(lg, dim=-1))
    return torch.cat(outs, dim=1)


class HFBackend:
    """LogprobBackend for HF causal LMs.

    Context is [BOS if the tokenizer has one] + tok(p0) + tok(p), no chat template;
    p0 and p are tokenized separately. Batches are right-padded (causal attention
    makes padding irrelevant for real tokens) and sorted by length.
    """

    def __init__(
        self,
        spec: ModelSpec,
        device: str = "cuda",
        batch_size: int = 8,
        attn_implementation: str = "sdpa",
        max_tokens: int | None = None,
        model: Any = None,
        fp32_head: bool = True,
        tokenizer: Any = None,
    ):
        self.spec = spec
        self.device = device
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.fp32_head = fp32_head
        self._tokenizer = tokenizer if tokenizer is not None else load_tokenizer(spec)
        self.model = model if model is not None else load_model(spec, device, attn_implementation)
        # both are slow for large vocabularies: memoize them on the (possibly pooled) tokenizer
        tok = self._tokenizer
        if not hasattr(tok, "_onepunch_cache"):
            tok._onepunch_cache = (SpanTokenizer(tok), hashing.tokenizer_hash(tok))
        self._spans, self._tok_hash = tok._onepunch_cache

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    def context_ids(self, prefix: bytes) -> list[int]:
        """[BOS] + tok(p0): the part of the context before the first token of p."""
        bos = self._tokenizer.bos_token_id
        ids = ([bos] if bos is not None else []) + (self._spans.encode(prefix) if prefix else [])
        if not ids:
            raise ValueError("empty context: the model has no BOS token, so p0 must be non-empty")
        return ids

    @torch.no_grad()
    def token_logprobs(self, prompts: Sequence[bytes], prefix: bytes = b"\n") -> list[TokenLogprobs]:
        if self.model is None:
            raise RuntimeError("backend is closed")
        if not all(prompts):
            raise ValueError("empty prompt")
        ctx = self.context_ids(prefix)
        encoded = [self._spans.encode_with_spans(p) for p in prompts]
        seqs = [ctx + ids for ids, _ in encoded]
        if self.max_tokens is not None and seqs and max(map(len, seqs)) > self.max_tokens:
            raise ValueError(f"context of {max(map(len, seqs))} tokens exceeds max_tokens={self.max_tokens}")
        pad = self._tokenizer.pad_token_id
        pad = 0 if pad is None else pad
        out: list[TokenLogprobs | None] = [None] * len(prompts)
        order = sorted(range(len(prompts)), key=lambda i: -len(seqs[i]))
        for k in range(0, len(order), self.batch_size):
            idx = order[k : k + self.batch_size]
            T = max(len(seqs[i]) for i in idx)
            ids = torch.full((len(idx), T), pad, dtype=torch.long)
            mask = torch.zeros((len(idx), T), dtype=torch.long)
            for r, i in enumerate(idx):
                ids[r, : len(seqs[i])] = torch.tensor(seqs[i])
                mask[r, : len(seqs[i])] = 1
            lp = sequence_logprobs(self.model, ids.to(self.device), mask.to(self.device),
                                   fp32_head=self.fp32_head).cpu().numpy()
            for r, i in enumerate(idx):
                s, L = len(ctx), len(seqs[i])
                out[i] = TokenLogprobs(
                    logprobs=lp[r, s - 1 : L - 1],
                    spans=encoded[i][1],
                    n_bytes=len(prompts[i]),
                    tokenizer_hash=self._tok_hash,
                )
        return out  # type: ignore[return-value]

    def close(self) -> None:
        """Drop the model reference and free cached GPU memory (a pool keeps its own reference)."""
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
