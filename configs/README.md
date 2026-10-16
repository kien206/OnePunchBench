# Configs

All entry points (`train.py`, `predict.py`, `experiments/*.py`) load `config.yaml` with Hydra. It picks one file from each **group** folder below and adds a few top-level blocks. Change anything from the command line:

```bash
python train.py search=random_swap                 # pick another file in a group
python train.py search.loop.steps=500 seed=3       # change one value
python train.py search.optimizer.n_candidates=64 episodes.n_episodes=16 readout.lambda0=0.01
python train.py -m seed=0,1,2 features=chunk1,chunk16    # sweep (multirun): 6 runs
python train.py --cfg job                          # print the composed config and exit
```

Each run saves its full resolved config to `runs/<date>/<time>_<script>/.hydra/config.yaml`, and the trained artifact also keeps a copy.

## Where the "training settings" are

This method has **no learning rate and no gradient-descent optimizer** (formulation, Bilevel section):
- The readout is ridge regression, solved in closed form.
- The prompt is optimized by **discrete search**: propose candidate prompts, score each exactly, keep the best.

The knobs that play the role of "optimizer, learning rate, epochs, early stopping, regularization":

| What you'd call it | Where | Keys |
| --- | --- | --- |
| Optimizer (prompt search algorithm) | `search/*.yaml` → `optimizer` | `_target_` picks the class. `random_swap`: `n_candidates`, `allowed` tokens, surrogate `tokenizer`. `best_of_n`: `generator`, `length_bytes`, `n_candidates`. Future GCG adds `top_k` and the surrogate groups |
| "Step size" | `search.optimizer` | Candidates per step (`n_candidates`). Each candidate is a single-token swap, so steps are small by design |
| Epochs / iterations | `search.loop` | `steps` |
| Early stopping | `search.loop` | `monitor_every` (check the monitor loss every N steps) and `patience` (stop after N checks without improvement) |
| Batch / minibatch | `episodes/*.yaml` | `n_episodes` per step, `n_in`/`n_out` models per episode, `step_families` × `step_models_per_family` models loaded per step, `redraw_pool_every` |
| Regularization | `readout/ridge.yaml` | `lambda0` (λ = λ0·N) and the `cv.lambda0_grid` searched when `train.cv_lambda0=true` |
| Model size / capacity | `features/*.yaml` | `d` (chunks = features), `l_chunk` (bytes per chunk, so \|p\| ≥ d·l_chunk) |
| Search constraints | `search.constraints` | `max_bytes` (L_in) and `max_tokens` (n_max) |
| Data | `pool/*.yaml`, `roles`, `scores` | Which models are loaded, what each split is used for, which benchmarks are predicted |
| Initialization | `train.init_prompt` | `data/prompts/<name>.txt` |
| Seed | `seed` | Episodes, candidates and bootstrap all derive from it |

If a readout trained by gradient descent is added later (e.g. the MLP ablation), its learning rate, optimizer and epochs go in **its own** readout file, e.g. `readout/mlp.yaml`:

```yaml
_target_: onepunch.readout.mlp.MLPReadout   # subclass of onepunch.readout.base.Readout
hidden: 64
lr: 1.0e-3
weight_decay: 1.0e-4
epochs: 500
optimizer: adam
```

Select it with `python train.py readout=mlp readout.lr=3e-4`. Each file holds the settings of the component it builds. No global "lr" exists, because the components that need one differ.

## Groups

| Group | Files | Built by | Contents |
| --- | --- | --- | --- |
| `backend/` | `hf.yaml` | `FeatureExtractor.from_cfg` | Model loading. `dtype` defaults to **float32**: E0 showed bf16 features depend on batch composition. Also `batch_size`, `attn_implementation`, and `pool_ram_gb` (pinned host RAM for the model cache) |
| `features/` | `chunk1`, `chunk16`, `chunk64` | `FeatureExtractor` | `d`, `l_chunk`, the prefix `p0`, and `persist` (disk cache of log-probs) |
| `readout/` | `ridge.yaml` | `onepunch.readout.build_readout` | Readout class plus its hyperparameters. `cv` is used only for λ0 selection |
| `episodes/` | `default.yaml` | `EpisodeSampler.from_cfg`, `build_step_pool_cfg` | Inner/outer sizes, per-family cap, ρ (tokenizer-disjoint fraction), episodes per step, step model pool, σ-standardization |
| `search/` | `none`, `random_swap`, `best_of_n` | `onepunch.search.build_optimizer` | `optimizer` (null = no search; train only the readout on the initial prompt), `loop`, `constraints` |
| `pool/` | `smoke`, `tier_s`, `tier_r` | `onepunch.data.select_pool` | **Which models are loaded**, nothing else: `model_list` (file(s) in `data/models/`), `model_ids` (inline) and `where` (conditions on registry columns, i.e. `Registry.filter` arguments such as `max_params_b`, `on_hub`, `max_per_family`). Unknown keys are an error |
| `logger/` | `wandb.yaml` | `onepunch.utils.wandb_logger.Logger` | `project`, `entity`, `group`, `tags`, and `mode`: `online`, `offline` or `disabled` |

## Models, splits and roles: how a run decides what to use

Three things, each in one place:

1. **Pool** (`pool/*.yaml`): which models are loaded at all.
2. **Split** of each model (column of `data/registry/registry.parquet`): assigned once per *family* by `scripts/assign_splits.py`, using the rules in `data/splits.yaml`:

   | Split | Meaning |
   | --- | --- |
   | `train` | Archive families: search episodes and the final readout |
   | `monitor` | Archive families: early stopping and choosing the best prompt only |
   | `calib` | Archive families: error bars only |
   | `testA` | Held-out archive families of the same era |
   | `testB` | Families released after the leaderboard closed (2025-03-13). They need our own lm-eval scores, so this split is **empty in stage 1** |
   | `testC` | Held-out architectures (MoE, hybrid, recurrent) |
   | `excluded` | Unusable: mixed-root merges, unsupported architectures, families with fewer than 3 models |

3. **Roles** (`roles:` in `config.yaml`): which splits of the pool train, monitor and evaluate. Example: `roles.eval=[testA]`.

Models without a score on every benchmark in `scores.benchmarks` are dropped. To see the result for any config before running it:

```bash
python scripts/describe_run.py pool=tier_s            # models/families per role, weights cached or not
python scripts/describe_run.py pool=tier_s --models   # ... and every model id
```

## Top-level blocks in `config.yaml`

- **`scores`**: what is predicted.
  - `benchmarks`: the six OLL v2 benchmarks; keys of `data/benchmarks.yaml`.
  - `sources: [archive]`: ground truth only from the Open LLM Leaderboard archive (stage 1).
  - `protocol_version`: scores from different evaluation protocols are never mixed.
- **`roles`**: see above.
- **`train`**: `init_prompt`, `cv_lambda0`, `n_boot`.
- **`predict`**: `artifact` (required), plus `model_ids` (any HF ids) or `splits` of the pool.
- **`paths`**: data locations. Override the root with `ONEPUNCH_ROOT`.
- Experiment scripts add their own options under `exp` (see each script's `EXP_DEFAULTS`). Set them with a leading `+`, e.g. `+exp.splits=[train,monitor]`.

## Adding a component

1. Write the class, e.g. `src/onepunch/search/evo.py` implementing `propose(state) -> list[bytes]`.
2. Add `configs/search/evo.yaml` with `optimizer._target_: onepunch.search.evo.Evolutionary` and its arguments, plus `loop` and `constraints`. Copy these from `random_swap.yaml`.
3. Run `python train.py search=evo`.

Readouts (`readout/`), pools (`pool/`) and feature settings (`features/`) work the same way.

Secrets (`HF_TOKEN`, `WANDB_API_KEY`) never go in configs. They go in the repo's `.env`; copy it from `.env.example`.
