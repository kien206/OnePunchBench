# data/

Models and scores are **data**, not config. Everything a run reads lives here.

| Path | Stage | What it is | Made by | Tracked in git |
| --- | --- | --- | --- | --- |
| `benchmarks.yaml` | 1 | The six Open LLM Leaderboard v2 benchmarks: lm-eval task, metric, shots, item count | hand-written | yes |
| `splits.yaml` | 1 | Rules for assigning each model *family* to a split (train / monitor / calib / testA / testB / testC / excluded) | hand-written | yes |
| `models/*.yaml` | 1 | Lists of HF model ids. Pools point to them via `model_list` | `scripts/select_pool_models.py` or by hand | yes |
| `prompts/*.txt` | 1 | Fixed prompts (raw bytes): the starting prompt and the T1 baselines | hand-written | yes |
| `archive/oll_v2_contents.parquet` | 1 | Raw snapshot of the Open LLM Leaderboard v2 (HF dataset `open-llm-leaderboard/contents`), as downloaded | `scripts/pull_open_llm_leaderboard.py` | no |
| `registry/registry.parquet` | 1 | **One row per model**: family, tokenizer hash, params, architecture, pinned revision, split, surrogate group | `pull_open_llm_leaderboard.py`, then `hf_to_registry.py`, then `assign_splits.py` | no |
| `registry/hub_parents.json` | 1 | Cache of parent links from Hub model cards, used to resolve families | `hf_to_registry.py` | no |
| `scores/scores.parquet` | 1 (+2) | **The ground truth that training and evaluation read.** Long format: `model_id, revision, benchmark, score, stderr, n_items, source, protocol_version, date`. Stage 1 has only `source=archive` rows | `pull_open_llm_leaderboard.py`; stage 2 adds `source=ours` via `run_groundtruth.py` | no |
| `cache/logprobs/` | 1 | Per-token log-probs of each (model, prompt), so features for any `d` need no new forward pass | `train.py`, `predict.py`, experiments | no |
| `lmeval_runs/` | **2** | Raw lm-eval outputs (`<model>/**/results_*.json`) from our own runs of the leaderboard protocol. **Empty until stage 2** | `scripts/run_groundtruth.py` | no |

## Pipeline

```bash
python scripts/pull_open_llm_leaderboard.py      # archive/ -> registry + scores (source=archive)
python scripts/hf_to_registry.py data/models/tier_s_round1.yaml   # Hub metadata + tokenizer hashes
python scripts/assign_splits.py                  # family -> split, using splits.yaml
python scripts/describe_run.py pool=tier_s       # check what a run would use
# stage 2 only, after stage 1 works:
python scripts/run_groundtruth.py --pool <pool>  # lm-eval -> lmeval_runs/ -> scores (source=ours)
python scripts/check_reproduction.py             # Gate 0: ours vs archive
```

The ground truth used is set in `configs/config.yaml` → `scores.sources` (stage 1: `[archive]`).
