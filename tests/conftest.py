"""Shared test fixtures: a synthetic model population with a fake feature source.

The population has latent abilities z_M (dim 3). Scores are a fixed squashed
linear map of z_M; features are a prompt-dependent linear map of z_M plus
small noise. So a readout CAN predict scores, and how well depends on the
prompt -- enough to test ridge, episodes, LOFO and the search loop without
loading any LLM.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from onepunch.types import REGISTRY_COLUMNS, SCORE_COLUMNS, ModelSpec

BENCHMARKS = ["ifeval", "bbh", "math_lvl5", "gpqa", "musr", "mmlu_pro"]
N_FAMILIES = 8
MODELS_PER_FAMILY = 5
TOKENIZERS = ["tokA", "tokB", "tokC"]
LATENT = 3


def _seed(*parts: object) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


@dataclass
class SyntheticPopulation:
    specs: list[ModelSpec]
    families: np.ndarray       # [N] str
    tokenizers: np.ndarray     # [N] str
    latent: np.ndarray         # [N, LATENT]
    S: np.ndarray              # [N, K] scores in [0, 1]
    benchmarks: list[str]
    registry: pd.DataFrame     # REGISTRY_COLUMNS
    scores: pd.DataFrame       # SCORE_COLUMNS (long format)


class FakeFeatureSource:
    """Implements onepunch.types.FeatureSource deterministically."""

    def __init__(self, population: SyntheticPopulation, d: int = 8, noise: float = 0.05):
        self.d = d
        self.noise = noise
        self._latent = {s.model_id: z for s, z in zip(population.specs, population.latent)}
        self.calls: list[tuple[int, int]] = []  # (n_models, n_prompts) per call

    def features(self, models: Sequence[ModelSpec], prompts: Sequence[bytes]) -> np.ndarray:
        self.calls.append((len(models), len(prompts)))
        out = np.empty((len(models), len(prompts), self.d), dtype=np.float64)
        for j, p in enumerate(prompts):
            A = np.random.default_rng(_seed("prompt", p)).normal(size=(LATENT, self.d))
            for i, m in enumerate(models):
                eps = np.random.default_rng(_seed("noise", m.model_id, p)).normal(size=self.d)
                out[i, j] = self._latent[m.model_id] @ A + self.noise * eps
        return out


def make_population(seed: int = 0) -> SyntheticPopulation:
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(LATENT, len(BENCHMARKS)))
    specs, fams, toks, lat, reg_rows = [], [], [], [], []
    for f in range(N_FAMILIES):
        fam = f"fam{f}"
        tok = TOKENIZERS[f % len(TOKENIZERS)]
        center = rng.normal(size=LATENT)
        for m in range(MODELS_PER_FAMILY):
            mid = f"org{f}/model{f}-{m}"
            z = center + 0.3 * rng.normal(size=LATENT)
            params = float(np.round(0.5 * (m + 1), 2))
            specs.append(ModelSpec(model_id=mid, params_b=params, family=fam, tokenizer_hash=tok))
            fams.append(fam)
            toks.append(tok)
            lat.append(z)
            reg_rows.append({
                "model_id": mid, "revision": "main", "family": fam, "root_model": f"org{f}/model{f}-0",
                "parents": [] if m == 0 else [f"org{f}/model{f}-0"], "tokenizer_hash": tok,
                "vocab_size": 32000, "params_b": params, "arch": "LlamaForCausalLM", "moe": False,
                "merged": False, "release_date": "2024-06-01", "dtype": "bfloat16", "loader": "hf",
                "trust_remote_code": False, "chat_template": False, "on_hub": True, "tier": "S", "split": "train",
                "surrogate_group": tok if f < 2 else None,
            })
    latent = np.stack(lat)
    S = 1.0 / (1.0 + np.exp(-(latent @ B) / 2.0))
    score_rows = [
        {"model_id": s.model_id, "revision": "main", "benchmark": b, "score": float(S[i, k]),
         "stderr": 0.01, "n_items": 1000, "source": "archive", "protocol_version": "oll_v2",
         "date": "2024-06-01"}
        for i, s in enumerate(specs) for k, b in enumerate(BENCHMARKS)
    ]
    return SyntheticPopulation(
        specs=specs, families=np.array(fams), tokenizers=np.array(toks), latent=latent, S=S,
        benchmarks=list(BENCHMARKS),
        registry=pd.DataFrame(reg_rows, columns=list(REGISTRY_COLUMNS)),
        scores=pd.DataFrame(score_rows, columns=list(SCORE_COLUMNS)),
    )


@pytest.fixture
def population() -> SyntheticPopulation:
    return make_population()


@pytest.fixture
def fake_source(population: SyntheticPopulation) -> FakeFeatureSource:
    return FakeFeatureSource(population)
