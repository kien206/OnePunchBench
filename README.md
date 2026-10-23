# OnePunchBench

One prompt `p`, one forward pass per model: byte-chunk log-probabilities of `p`, passed through a ridge readout, predict a model's benchmark scores. Method: `formulation_draft.txt` (bilevel section). Experiment plan: `initial_plan.md`. Code plan: `docs/plan_milestone1.md`.

## Setup

```bash
PY=/media/volume/DoubleSteeringLLM/miniconda3/envs/onepunch/bin/python   # main env
cp .env.example .env        # fill in HF_TOKEN (gated Llama/Gemma) and optionally WANDB_API_KEY
$PY -m pytest -q            # -m "not slow" skips model/tokenizer downloads
```

`.env` is loaded automatically on `import onepunch`, and variables already exported in the shell take precedence. It sets `HF_HOME` to the data volume, because the root disk is small. Set `HF_HUB_OFFLINE=1` there once the weights are cached.

## Train and predict

```bash
./run_train.sh                                           # train / validate on val / test, prints RMSE (settings at the top of the file)
nohup ./run_train.sh > run_train.out 2>&1 &              # same, survives logout
$PY scripts/describe_run.py pool=tier_s                  # what a run would use (models per role, cached weights)
$PY scripts/report_metrics.py runs/<date>/<time>_train   # RMSE / Spearman table of a finished run
$PY train.py                                             # smoke pool, fixed prompt, readout only
$PY train.py search=random_swap search.loop.steps=200    # prompt search (Algorithm 1 loop)
$PY train.py pool=tier_s search=random_swap logger.mode=offline
$PY predict.py predict.artifact=runs/<date>/<time>_train/artifact predict.model_ids=[Qwen/Qwen2.5-0.5B]
$PY predict.py predict.artifact=... pool=tier_s predict.splits=[testA]   # adds metrics where scores exist
```

- **Training** writes `runs/<date>/<time>_train/artifact/`: `prompt.txt` (p*), `predictor.npz` (W*, feature and score means, σ, d, p0, dtype), `config.yaml` and `train_models.txt`. It then evaluates on the splits in `roles.eval`.
- **Prediction** needs only the artifact. Any HF model id works, whether or not it is in the registry.
- **All settings**, and where the "optimizer / learning rate / epochs" knobs live, are explained in [`configs/README.md`](configs/README.md).

## Data (stage 1: Open LLM Leaderboard only)

What every folder in `data/` holds, and which stage uses it: [`data/README.md`](data/README.md).

Models and scores are data, not config.
- **Models**: `data/registry/registry.parquet`, one row per model. A pool (`configs/pool/*.yaml`) selects rows.
- **Scores**: `data/scores/scores.parquet`, a long table with columns `model_id, revision, benchmark, score, stderr, n_items, source, protocol_version, date`.
- **Splits** are per family and stored in the registry, not the score table.

Stage 1 uses only the six Open LLM Leaderboard v2 benchmarks and the models scored there (`scores.sources: [archive]`). Every model in the smoke (12), tier_s (73) and tier_r (629) pools has archive scores.

```bash
$PY scripts/pull_open_llm_leaderboard.py [--refresh]   # archive -> registry + scores (source=archive)
$PY scripts/hf_to_registry.py data/models/smoke.yaml   # Hub metadata, tokenizer hash, pinned revision
$PY scripts/assign_splits.py                           # family -> split (data/splits.yaml)
$PY scripts/download_models.py --pool smoke            # weights (safetensors only)
```

**Stage 2 (only after stage 1 works): our own ground truth with lm-eval**, for models released after the leaderboard closed. lm-eval lives in a separate env (`onepunch-eval`), because its math extra conflicts with Hydra. It is called by subprocess.

```bash
$PY scripts/run_groundtruth.py --pool smoke --dry-run  # OLL v2 protocol -> scores (source=ours)
$PY scripts/check_reproduction.py                      # Gate 0: ours vs archive within 2 SE
# then: scores.sources=[ours,archive]
```

## Experiments

```bash
$PY experiments/e0_feature_checks.py logger.mode=offline                  # E0 determinism checks
$PY experiments/t1_baselines.py pool=tier_s +exp.splits=[train,val]   # T1 baselines, Gate 1
```

Each experiment is a standalone script. Its own options are `EXP_DEFAULTS`, overridden with `+exp.<key>=...`.

## Layout

| Path | What |
| --- | --- |
| `train.py`, `predict.py` | Entry points (thin; the logic is in `src/onepunch/pipeline.py`) |
| `configs/` | Hydra configs; see `configs/README.md` |
| `experiments/` | One script per experiment |
| `scripts/` | Data and infra utilities |
| `src/onepunch/types.py` | Shared contracts: table schemas, `ModelSpec`, `TokenLogprobs`, `FeatureSource`, `Episode`, `SearchState`, `Optimizer` |
| `src/onepunch/data/` | `Registry`, `ScoreTable`, benchmarks, splits, archive census |
| `src/onepunch/features/` | HF backend (`[BOS]+tok(p0)+tok(p)`, byte spans), chunk features, log-prob cache, model pool, `FeatureExtractor` |
| `src/onepunch/readout/` | `Readout` ABC and `RidgeReadout` (closed form, float64, differentiable) |
| `src/onepunch/episodes/` | Episode sampler, episodic and validation loss, λ0 CV |
| `src/onepunch/search/` | `SearchLoop`, objectives, constraints, `RandomSwap`, `BestOfN` |
| `src/onepunch/eval/` | Metrics, family bootstrap, LOFO, `FittedPredictor` |
| `src/onepunch/groundtruth/` | lm-eval wrapper and reproduction check (stage 2) |

## Extending

- **New model** (stage 1: must be in the leaderboard archive). Add its id to a `data/models/*.yaml`, then run `hf_to_registry.py` and `assign_splits.py`.
- **New benchmark.** Add an entry to `data/benchmarks.yaml` and append score rows to `scores.parquet`.
- **New optimizer, readout or pool.** Add one class and one YAML; see `configs/README.md`. Gradient methods (GCG) also implement `before_step(objective, state)`, which receives the step's episodes, models, scores, σ and readout.
- **New experiment.** Write a script in `experiments/` using `onepunch.utils.experiment.{start_run, load_population, load_prompt}` and/or `onepunch.pipeline`.

## Conventions and findings so far

- **Prompts are raw bytes and must be valid UTF-8 and NFC.** Qwen and Pythia tokenizers NFC-normalize their input. `PromptConstraints` enforces both.
- **Features are float32 by default.** In E0 on the smoke pool, bf16 features moved by 3.1% of the between-model spread when batch composition changed; fp32 moved by 0.001%.
- **Gated repos (Llama, Gemma) need `HF_TOKEN` in `.env`.** Without it, the pools use ungated fine-tunes.
