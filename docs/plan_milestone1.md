# OnePunchBench — code plan, milestone 1 (E0 + T1 core, search stubbed)

## Context
The paper predicts a model's benchmark scores from its log-probabilities on one fixed prompt `p`, using a readout fitted across models (`formulation_draft.txt`, bilevel section). `initial_plan.md` lays out the experiments E0 → T1 (gate) → T2/T3 search → T4/V1–V9. The repo is empty. We need a codebase that is:
- easy to extend: a new model is a registry row, a new benchmark is score-table rows, a new optimizer or readout is one class plus one YAML;
- configured by Hydra, with wandb tracking;
- simple and modular.

Milestone 1 builds everything needed for Gate 0 and Gate 1. It also builds the search-loop interface with trivial optimizers (random swap, best-of-N), so T2 and T3 later only add `propose()` classes.

Decisions already made:
- Hydra and wandb.
- E0+T1 first.
- The lm-eval wrapper lives in this repo.
- Experiments are **custom scripts**, not config groups.
- The readout is **pluggable** (ridge now, MLP or others later).
- Utility scripts go in a top-level `scripts/` folder.
- The initial pool is **reduced** to fit the hardware.
- Implementation is done by **at most 3 Sonnet subagents with disjoint file ownership**.

## Design principles
1. **Models and scores are data, not config.** One `registry.parquet` is built by `scripts/`. Code selects models with filters and never lists them.
2. **The score table is the unified benchmark format.** It is long-format and holds no split information. Splits are per family and live in the registry.
3. **Cache raw per-token logprobs, not features.** Chunk features for any `d` are computed from cached `(logprobs, byte_spans)`. So the d=1/16/64 ablations need no new forward passes.
4. **Loop order is model-major.** Load each model once and score all prompts/candidates in batch.
5. **Component configs, script experiments.** `configs/` holds only reusable component settings: backend, features, readout, episodes, search, pool, logger. Each experiment is a Python script in `experiments/` that composes library pieces explicitly, using `@hydra.main(config_path="../configs")` for settings and CLI overrides.
6. **Swappable components** are built with `hydra.utils.instantiate(_target_=...)`: backend, readout, episode sampler, optimizer.

## Repository layout
```
pyproject.toml                    # package `onepunch` (src layout), extras: [lmeval], [dev]
configs/
  config.yaml                     # defaults: backend=hf features=chunk16 readout=ridge episodes=default search=none pool=smoke logger=wandb; seed; paths
  backend/   hf.yaml
  features/  chunk1.yaml chunk16.yaml chunk64.yaml      # d, L_chunk, prefix p0
  readout/   ridge.yaml                                  # _target_, lambda0 grid, standardize; (mlp.yaml later)
  episodes/  default.yaml                                # N_in, N_out, N_fam, E, rho
  search/    none.yaml random_swap.yaml best_of_n.yaml
  pool/      smoke.yaml tier_s.yaml tier_r.yaml          # registry filters + pool_ram_gb
  logger/    wandb.yaml                                  # project, entity, mode: online|offline|disabled
experiments/                      # research runs, one script each, free-form
  e0_feature_checks.py            # determinism, batch-size invariance, chunk-sum check on real models
  t1_baselines.py                 # LOFO for baselines 1–6, metrics + family bootstrap -> wandb
(search_smoke.py was replaced by train.py / predict.py at the repo root)
scripts/                          # data/infra utilities (not experiments)
  hf_to_registry.py               # list of HF ids (YAML/txt) -> registry rows (Hub metadata, tokenizer hash, params, base_model tree)
  pull_open_llm_leaderboard.py    # OLL v2 archive -> registry rows + scores.parquet (source=archive)
  assign_splits.py                # family -> split, surrogate groups; writes registry
  run_groundtruth.py              # lm-eval leaderboard protocol -> scores.parquet (source=ours)
  check_reproduction.py           # Gate 0: our scores vs archive within 2 SE
  download_models.py              # prefetch/evict model weights for a pool
data/ (gitignored): registry/registry.parquet, models/*.yaml (HF id lists), scores/scores.parquet,
                    benchmarks.yaml, prompts/*.txt, cache/logprobs/<model>@<rev>/<sha(p)>_<sha(p0)>.npz
src/onepunch/
  types.py           # SHARED CONTRACTS (written before dispatch, see below)
  data/      registry.py census.py scores.py benchmarks.py splits.py
  groundtruth/ lmeval_runner.py reproduce.py
  features/  hf_backend.py chunking.py cache.py extractor.py model_pool.py
  readout/   base.py ridge.py           # Readout ABC + RidgeReadout; MLP drops in later
  episodes/  sampler.py loss.py cv.py   # family/tokenizer-disjoint episodes; episodic loss over any Readout
  search/    loop.py constraints.py random_swap.py best_of_n.py
  eval/      metrics.py bootstrap.py lofo.py
  utils/     wandb_logger.py seed.py hashing.py
tests/  test_data_*.py  test_features_*.py  test_readout_*.py  test_episodes_*.py  test_search_*.py  test_eval_*.py
```

## Shared contracts (`src/onepunch/types.py` + `readout/base.py`, written by me first)
```python
@dataclass class TokenLogprobs:  logprobs: np.ndarray  # [n] float32
                                 spans: np.ndarray     # [n,2] int32 byte [start,end) within p
                                 n_bytes: int; tokenizer_hash: str
class LogprobBackend(Protocol):  # one loaded model
    def token_logprobs(self, prompts: list[bytes], prefix: bytes) -> list[TokenLogprobs]
    def tokenizer(self) -> "PreTrainedTokenizerBase"
class FeatureSource(Protocol):   # what search/eval consume
    def features(self, model_ids: list[str], prompts: list[bytes]) -> np.ndarray  # [N, P, d]
class Readout(ABC):
    closed_form: bool            # True -> differentiable fit usable in bilevel hypergradient
    def fit(self, Phi: Tensor, S: Tensor) -> "Readout"     # Phi/S already centered/standardized
    def predict(self, Phi: Tensor) -> Tensor
@dataclass class Episode: in_idx: np.ndarray; out_idx: np.ndarray; tokenizer_disjoint: bool
class Optimizer(Protocol):
    def propose(self, state: "SearchState") -> list[bytes]
```
- Centering and σ-standardizing live in `episodes/loss.py` and `eval/lofo.py`, outside the readout. So any readout sees the same normalized inputs.
- `RidgeReadout` picks the primal (d×d) or dual (N×N) form, runs in float64, and uses λ = λ0·N_in.
- An MLP readout later just sets `closed_form=False`. The episode loss still works, and the hypergradient path checks the flag.

## Key data schemas
- **registry.parquet**: `model_id, revision, family, root_model, parents, tokenizer_hash, vocab_size, params_b, arch, moe, merged, release_date, dtype, loader, trust_remote_code, on_hub, tier (S|R), split, surrogate_group`.
- **scores.parquet** (long format): `model_id, revision, benchmark, score, stderr, n_items, source (archive|ours), protocol_version, date`. `ScoreTable.matrix(model_ids, benchmarks)` returns `(S, mask)`.
- **Logprob cache npz**: the `TokenLogprobs` fields plus `backend, dtype, transformers_version`. The cache also has an in-memory-only mode for search candidates.

## Reduced initial pool (hardware: 1×A100 40GB, 115 GB RAM, ~403 GB free disk)
- **Smoke pool** (about 8 models, ≤ 0.5B): SmolLM2-135M/360M, Qwen2.5-0.5B (+ 1–2 fine-tunes), and similar models with archive scores. Used for tests and CI-style runs.
- **Tier S, round 1**: about 60–80 models ≤ 2B, roughly 150 GB on disk. Families: Qwen2.5 (0.5/1.5), Llama 3.2-1B, SmolLM2, Gemma-2-2B, and their archive fine-tunes, with at most 6 models per family.
- **Step pool**: 24 models (8 families × 3) instead of 48, so the pinned RAM cache stays ≤ 60 GB. Default `pool_ram_gb: 60`.
- **Tier R**: streamed on demand (download → score all pending prompts → delete). It is never held in full.

## Implementation: dispatch (max 3 Sonnet subagents, disjoint file ownership)
**Step 0 (me, before dispatch).** Write these, then freeze them:
- `pyproject.toml`
- package `__init__.py`s
- `src/onepunch/types.py`, `readout/base.py`
- `configs/config.yaml`, `configs/logger/wandb.yaml`
- `utils/{seed,hashing,wandb_logger}.py`
- `tests/conftest.py`, with a fake `FeatureSource` and a tiny synthetic registry/scores fixture

Agents may *read* but not edit these. If an agent needs a contract change, it reports that instead of editing.

| Agent | Owns (exclusive write) | Delivers |
| --- | --- | --- |
| **A: data and ground truth** | `src/onepunch/data/**`, `src/onepunch/groundtruth/**`, `scripts/**`, `configs/pool/**`, `data/benchmarks.yaml`, `data/models/*.yaml`, `tests/test_data_*.py` | Registry, census, ingest of the OLL v2 archive (`open-llm-leaderboard/contents`), split assignment, ScoreTable, lm-eval wrapper, reproduction check, the reduced pool lists |
| **B: features** | `src/onepunch/features/**`, `configs/backend/**`, `configs/features/**`, `tests/test_features_*.py` | HF backend (BOS + tok(p0) + tok(p), no chat template, incremental-decode byte spans), chunking (byte-weighted ω), logprob cache, model-major extractor implementing `FeatureSource`, model pool with pinned-RAM LRU |
| **C: readout, episodes, search, eval** | `src/onepunch/readout/ridge.py`, `src/onepunch/episodes/**`, `src/onepunch/search/**`, `src/onepunch/eval/**`, `configs/{readout,episodes,search}/**`, `tests/test_{readout,episodes,search,eval}_*.py` | RidgeReadout, episode sampler (family-disjoint, per-family cap, ρ tokenizer-disjoint, active group), episodic loss, λ0 family-grouped CV, SearchLoop plus constraints plus random_swap and best_of_n, metrics, family bootstrap, LOFO. Tested against the fake `FeatureSource` |

**Step 2 (me, after all agents finish).**
- Write `experiments/*.py`.
- Run the full test suite and the smoke runs.
- Fix integration issues.
- Check that no agent edited outside its owned paths, by comparing the file list against the ownership table.

## Verification
- **Unit tests**:
  - chunk features × |c_j| sum to the total logprob (byte-level BPE and SentencePiece);
  - primal and dual ridge agree;
  - closed-form ∂ℓ/∂Φ_out and ∂ℓ/∂Φ_in match autograd in float64;
  - episodes are family-disjoint, respect the cap, and tokenizer-disjoint episodes are correct;
  - constraints reject invalid UTF-8, over-length prompts and round-trip failures;
  - bootstrap resamples families;
  - registry filters and split assignment are deterministic.
- **E0 smoke**: `python experiments/e0_feature_checks.py pool=smoke`. Features are identical across batch sizes and repeat runs, and noise is under 1% of the between-model spread.
- **T1 smoke**: `python experiments/t1_baselines.py pool=smoke logger.mode=offline`. Produces LOFO Spearman/RMSE with family-bootstrap CIs and logs them to wandb.
- **Search smoke**: `python train.py search=random_swap search.loop.steps=3`.
