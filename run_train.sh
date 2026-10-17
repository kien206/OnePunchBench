#!/usr/bin/env bash
# Train on the `train` split of a pool, validate (early-stop) on `monitor`, test on the unseen
# splits (roles.eval in configs/config.yaml), and print RMSE.
#
# Edit the settings below, then run:
#   ./run_train.sh
# Extra arguments are passed to train.py as Hydra overrides (they win over the settings below):
#   ./run_train.sh seed=1 features=chunk1
# To keep it running after you log out:
#   nohup ./run_train.sh > run_train.out 2>&1 &
#
# Output: runs/<date>/<time>_train/  (metrics.json, train.log, history.json, artifact/)
set -euo pipefail
cd "$(dirname "$0")"

# ----------------------------------------------------------------------------- settings

# Python of the project's conda env.
PY=/media/volume/DoubleSteeringLLM/miniconda3/envs/onepunch/bin/python

# Which models are loaded: a file in configs/pool/ (smoke = 12 tiny models, tier_s = search pool).
# Train / monitor / test membership comes from each model's split (data/splits.yaml) and `roles`.
POOL=tier_s

# Prompt search algorithm: a file in configs/search/.
#   random_swap  one random token swap per candidate, keep the best (gradient-free)
#   best_of_n    fresh random prompts every step
#   none         no search: fit the readout on the initial prompt only (monitor is not used)
SEARCH=random_swap

# Feature map f(model, prompt): a file in configs/features/.
#   chunk16      log-probs of the prompt averaged over 16 byte chunks (default method)
#   chunk1       one feature: bits per byte of the whole prompt
#   chunk64      64 chunks
#   log_params   log(parameter count), ignores the prompt (use with SEARCH=none)
FEATURES=chunk16

# Number of search steps (each step proposes candidates and keeps the best one).
STEPS=40

# Check the monitor loss every this many steps; the run returns the prompt with the best monitor loss.
MONITOR_EVERY=5

# Stop after this many monitor checks without improvement (early stopping).
PATIENCE=4

# Random seed (episodes, candidate swaps, bootstrap).
SEED=0

# Where results go.
RUN_DIR=runs/$(date +%Y-%m-%d)/$(date +%H-%M-%S)_train

# ----------------------------------------------------------------------------- run

ARGS=(
    pool=$POOL
    search=$SEARCH
    features=$FEATURES
    search.loop.steps=$STEPS
    search.loop.monitor_every=$MONITOR_EVERY
    search.loop.patience=$PATIENCE
    seed=$SEED
    "$@"
)

echo "=== what this run uses"
"$PY" scripts/describe_run.py "${ARGS[@]}" hydra.run.dir="$RUN_DIR/describe"

echo "=== training -> $RUN_DIR"
"$PY" train.py "${ARGS[@]}" hydra.run.dir="$RUN_DIR"

echo "=== test results"
"$PY" scripts/report_metrics.py "$RUN_DIR"
