"""Shared contracts between onepunch subpackages.

FROZEN: every subpackage codes against these definitions. Change them only in
coordination with all owners (data, features, readout/episodes/search/eval).

Notation follows formulation_draft.txt (bilevel section):
  p      prompt, stored as raw bytes
  p0     fixed prefix (default b"\\n"), tokenized separately, belongs to no chunk
  d      number of byte chunks = feature dimension
  K      number of benchmarks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np

# --------------------------------------------------------------------------
# Table schemas (data/registry.parquet and data/scores/scores.parquet)
# --------------------------------------------------------------------------

#: Columns of registry.parquet, one row per (model_id, revision).
REGISTRY_COLUMNS: tuple[str, ...] = (
    "model_id",          # HF repo id, e.g. "Qwen/Qwen2.5-0.5B-Instruct"
    "revision",          # HF commit sha or "main"
    "family",            # root base model of the lineage; the split unit
    "root_model",        # HF id of the root pretrained model
    "parents",           # list[str] of direct parents (fine-tune / merge sources)
    "tokenizer_hash",    # onepunch.utils.hashing.tokenizer_hash
    "vocab_size",        # int
    "params_b",          # float, billions of parameters
    "arch",              # HF config `architectures[0]` or archive Architecture
    "moe",               # bool
    "merged",            # bool
    "release_date",      # ISO date string (first Hub commit / archive submission)
    "dtype",             # "bfloat16" | "float16" | "float32"
    "loader",            # "hf" | "hf_vlm_text" | ... (see features backend)
    "trust_remote_code", # bool
    "chat_template",     # bool, lm-eval ground truth uses the chat template (archive column)
    "on_hub",            # bool, weights still downloadable
    "tier",              # "S" (search, <= ~2-4B) | "R" (readout/eval only)
    "split",             # "train" | "monitor" | "calib" | "testA" | "testB" | "testC" | "excluded"
    "surrogate_group",   # str | None, e.g. "qwen2" -- only for train models used for gradients
)

#: Columns of scores.parquet, long format: one row per (model, benchmark, source).
SCORE_COLUMNS: tuple[str, ...] = (
    "model_id",
    "revision",
    "benchmark",         # key in data/benchmarks.yaml, e.g. "bbh"
    "score",             # raw accuracy in [0, 1] under the fixed protocol
    "stderr",            # sampling standard error, NaN if unknown
    "n_items",           # int
    "source",            # "archive" | "ours"
    "protocol_version",  # e.g. "oll_v2" ; must match across sources to be mixed
    "date",              # ISO date the score was produced
)

SPLITS: tuple[str, ...] = ("train", "monitor", "calib", "testA", "testB", "testC", "excluded")


@dataclass(frozen=True)
class ModelSpec:
    """Everything the features backend needs to load one model.

    Produced by `onepunch.data.registry.Registry.spec(model_id)`, consumed by
    `onepunch.features`.
    """

    model_id: str
    revision: str = "main"
    dtype: str = "bfloat16"
    loader: str = "hf"
    trust_remote_code: bool = False
    params_b: float | None = None
    family: str | None = None
    tokenizer_hash: str | None = None


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


@dataclass
class TokenLogprobs:
    """Per-token log-probabilities of one model on one prompt p.

    Context convention: [BOS if any] + tok(p0) + tok(p), no chat template.
    Only the tokens of tok(p) are stored.
    """

    logprobs: np.ndarray  # [n] float32, natural-log P_M(u | everything before u)
    spans: np.ndarray     # [n, 2] int32, byte span [start, end) of token u within p;
                          # bytes a tokenizer adds on its own (dummy prefix space) are
                          # excluded, so a token may have start == end
    n_bytes: int          # len(p)
    tokenizer_hash: str

    def __post_init__(self) -> None:
        self.logprobs = np.asarray(self.logprobs, dtype=np.float32)
        self.spans = np.asarray(self.spans, dtype=np.int32).reshape(-1, 2)
        if self.logprobs.shape[0] != self.spans.shape[0]:
            raise ValueError("logprobs and spans must have the same length")


@runtime_checkable
class LogprobBackend(Protocol):
    """One loaded model that scores prompts (no generation)."""

    spec: ModelSpec

    def token_logprobs(self, prompts: Sequence[bytes], prefix: bytes) -> list[TokenLogprobs]:
        """Score each prompt in one forward pass (batched internally)."""
        ...

    @property
    def tokenizer(self) -> Any:  # transformers.PreTrainedTokenizerBase
        ...

    def close(self) -> None:
        """Release GPU memory."""
        ...


@runtime_checkable
class FeatureSource(Protocol):
    """What readout / episodes / search / eval consume.

    Implementations compute missing (model, prompt) pairs model-major
    (load each model once, score all its pending prompts) and cache them.
    """

    d: int

    def features(self, models: Sequence[ModelSpec], prompts: Sequence[bytes]) -> np.ndarray:
        """Return Phi of shape [N_models, N_prompts, d], float64."""
        ...


# --------------------------------------------------------------------------
# Episodes and search
# --------------------------------------------------------------------------


@dataclass
class Episode:
    """One inner/outer split of the step's models (indices into the step's model list)."""

    in_idx: np.ndarray
    out_idx: np.ndarray
    tokenizer_disjoint: bool = False


@dataclass
class SearchState:
    """Read-only view an Optimizer gets each step; `extras` carries optional
    inputs such as gradients, the active surrogate tokenizer, or episodes."""

    step: int
    prompt: bytes
    loss: float
    best_prompt: bytes
    best_loss: float
    rng: np.random.Generator
    history: list[dict[str, Any]] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Optimizer(Protocol):
    """Proposes candidate prompts. Scoring and selection are done by the SearchLoop."""

    def propose(self, state: SearchState) -> list[bytes]:
        ...
