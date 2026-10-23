"""Training and inference, behind the entry points train.py and predict.py.

Training = prompt search (configs/search) on episodes of the training families, then one
final readout fit (formulation, Algorithm 1). It writes an artifact directory:

    <run>/artifact/prompt.txt       p* as raw bytes
                   predictor.npz    W*, feature means f_bar, score means s_bar, sigma, meta (d, p0, dtype, ...)
                   config.yaml      resolved config of the training run
                   train_models.txt models the readout was fitted on

Inference needs only the artifact: score p* once per model, chunk, center, apply W*.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from onepunch.data.registry import Registry, select_pool
from onepunch.data.scores import ScoreTable
from onepunch.episodes import EpisodeSampler, build_step_pool_cfg, select_lambda0, standardize_scales
from onepunch.eval.bootstrap import family_bootstrap
from onepunch.eval.baselines import baseline_metrics
from onepunch.eval.lofo import FittedPredictor, fit_predictor
from onepunch.eval.metrics import mean_spearman, standardized_mse, summarize
from onepunch.features import build_feature_source
from onepunch.features.hf_backend import load_tokenizer
from onepunch.readout import build_readout, lambda0_grid
from onepunch.search import (EpisodicObjective, SearchLoop, ValidationObjective, build_constraints,
                             build_optimizer)
from onepunch.types import FeatureSource, ModelSpec
from onepunch.utils.experiment import Population, load_population, load_prompt, start_run

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- artifact

def save_artifact(out: Path, prompt: bytes, predictor: FittedPredictor, cfg: DictConfig,
                  train_models: Sequence[str]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompt.txt").write_bytes(prompt)
    predictor.save(out / "predictor.npz")
    OmegaConf.save(cfg, out / "config.yaml", resolve=True)
    (out / "train_models.txt").write_text("\n".join(train_models) + "\n")
    return out


def load_artifact(path: str | Path) -> tuple[bytes, FittedPredictor]:
    path = Path(path)
    return (path / "prompt.txt").read_bytes(), FittedPredictor.load(path / "predictor.npz")


def _feature_cfg_from_meta(cfg: DictConfig, meta: dict[str, Any]) -> DictConfig:
    """Copy of cfg whose features/backend settings match the ones the artifact was trained with."""
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if "features_cfg" in meta:   # the whole `features` group the artifact was trained with
        cfg.features = OmegaConf.create(meta["features_cfg"])
    else:                        # artifacts from before features._target_ existed
        cfg.features.d = int(meta["d"])
        cfg.features.prefix = meta["prefix"]
        cfg.features.l_chunk = int(meta.get("l_chunk", cfg.features.l_chunk))
    cfg.backend.dtype = meta.get("dtype", cfg.backend.dtype)
    return cfg


# --------------------------------------------------------------------------- evaluation

def evaluate(predictor: FittedPredictor, extractor: FeatureSource, pop: Population, prompt: bytes,
             n_boot: int, rng: np.random.Generator) -> tuple[dict[str, float], np.ndarray, Population]:
    """Metrics of the predictor on `pop` (which has ground truth); returns (metrics, predictions, scored).

    Models the backend cannot load (e.g. custom code without trust_remote_code) are skipped and
    counted in `n_failed`; metrics and `scored` cover the remaining models. `loss` is in the units
    of the search loss (squared error / training sigma)."""
    rows, ok = [], []
    for i, spec in enumerate(pop.specs):
        try:
            rows.append(extractor.features([spec], [prompt])[0, 0])
            ok.append(i)
        except Exception as e:
            log.error("skipping %s: %s: %s", spec.model_id, type(e).__name__, str(e).splitlines()[0][:200])
    n_failed = len(pop) - len(ok)
    pop = pop.subset(np.isin(np.arange(len(pop)), ok))
    if len(pop) == 0:
        raise RuntimeError("no model of this split could be scored")
    pred = predictor.predict(np.stack(rows))
    metrics = summarize(pred, pop.S, pop.families, pop.benchmarks)
    metrics["loss"] = standardized_mse(pred, pop.S, predictor.sigma)
    if len(set(pop.families)) > 1:
        _, lo, hi = family_bootstrap(mean_spearman, pred, pop.S, pop.families, n_boot=n_boot, rng=rng)
        metrics.update({"spearman/mean_ci_lo": lo, "spearman/mean_ci_hi": hi})
    metrics["n_models"], metrics["n_families"], metrics["n_failed"] = len(pop), len(set(pop.families)), n_failed
    return metrics, pred, pop


# --------------------------------------------------------------------------- training

def _cv_lambda0(readout, extractor: FeatureSource, pop: Population, prompt: bytes, grid) -> float:
    """Set readout.lambda0 by family-grouped CV; returns the held-out-family loss at that lambda0."""
    if not hasattr(readout, "lambda0"):
        return math.nan
    Phi = extractor.features(pop.specs, [prompt])[:, 0]
    readout.lambda0, table = select_lambda0(Phi, pop.S, pop.families, grid)
    log.info("lambda0 = %g (CV mse: %s)", readout.lambda0, {k: round(v, 4) for k, v in table.items()})
    return table[readout.lambda0]


def train(cfg: DictConfig) -> Path:
    """Search p* and fit W*; returns the artifact directory."""
    run = start_run(cfg, "train")
    t, roles = cfg.train, cfg.roles
    pop = load_population(cfg)
    tr, va = pop.where(roles.train), pop.where(roles.val or [])
    if len(set(tr.families)) < 2:
        raise SystemExit(f"need >= 2 training families, got {len(set(tr.families))} (pool={cfg.pool})")
    log.info("train: %d models / %d families; val: %d models", len(tr), len(set(tr.families)), len(va))

    init = load_prompt(cfg, t.init_prompt)
    persistent = build_feature_source(cfg)                # named prompts -> disk cache
    source = build_feature_source(cfg, persist=False,     # search candidates -> memory only;
                                  share_with=persistent)  # one model pool = one RAM budget
    readout, grid = build_readout(cfg.readout), lambda0_grid(cfg.readout)
    lofo = _cv_lambda0(readout, persistent, tr, init, grid) if t.cv_lambda0 else math.nan
    sigma = standardize_scales(tr.S, bool(cfg.episodes.standardize))

    optimizer = build_optimizer(cfg.search)
    history: list[dict] = []
    init_val = math.nan
    summary: dict[str, Any] = {"n_train_models": len(tr), "n_train_families": len(set(tr.families)),
                               "n_val_models": len(va), "lambda0/init": getattr(readout, "lambda0", None)}
    if optimizer is None:
        log.info("search=none: fixed-prompt training (readout only)")
        best = init
    else:
        objective = EpisodicObjective(source, tr.specs, tr.S, tr.families, tr.tokenizers,
                                      EpisodeSampler.from_cfg(cfg.episodes), readout, sigma,
                                      step_pool=build_step_pool_cfg(cfg.episodes))
        val = ValidationObjective(source, tr.specs, tr.S, va.specs, va.S, readout, sigma) if len(va) else None
        if val is None:
            log.warning("no validation models: the best prompt is chosen by step loss")
        reps = {tok: s for tok, s in zip(pop.tokenizers, pop.specs)}   # one model per tokenizer
        constraints = build_constraints(cfg.search, int(cfg.features.d) * int(cfg.features.get("l_chunk", 0)),
                                        [load_tokenizer(s) for s in reps.values()])
        loop = SearchLoop(optimizer, objective, val, constraints, logger=run.logger,
                          **OmegaConf.to_container(cfg.search.loop, resolve=True))
        result = loop.run(init, run.rng)
        best, history, init_val = result.best_prompt, result.history, result.init_val_loss
        summary.update({"val/init_loss": init_val, "val/best_loss": result.best_val_loss,
                        "train/best_step_loss": result.best_step_loss, "stopped_early": len(history) < loop.steps})
        if t.cv_lambda0:   # formulation: lambda0 re-selected for p*
            lofo = _cv_lambda0(readout, persistent, tr, best, grid)

    Phi = persistent.features(tr.specs, [best])[:, 0]
    meta = {"d": int(persistent.d), "features_cfg": OmegaConf.to_container(cfg.features, resolve=True),
            "prefix": str(cfg.features.get("prefix", "")), "l_chunk": int(cfg.features.get("l_chunk", 0)),
            "dtype": cfg.backend.dtype, "benchmarks": tr.benchmarks, "readout": cfg.readout.get("_target_"),
            "lambda0": getattr(readout, "lambda0", None), "score_sources": list(cfg.scores.sources)}
    predictor = fit_predictor(Phi, tr.S, readout, bool(cfg.episodes.standardize), meta=meta)
    art = save_artifact(run.out_dir / "artifact", best, predictor, cfg, tr.model_ids)
    init_rec = [] if math.isnan(init_val) else [{"step": -1, "val_loss": init_val}]  # step -1: initial prompt
    (run.out_dir / "history.json").write_text(json.dumps(init_rec + history, indent=1, default=float))

    summary.update({"prompt_changed": best != init, "steps_run": len(history),
                    "lambda0/final": getattr(readout, "lambda0", None), "train/lofo_loss": lofo})
    scored: dict[str, Population] = {}   # per split, the models the prompt predictor was scored on
    for split in dict.fromkeys(list(roles.val or []) + list(roles.eval or [])):   # val: cached by the search
        ev = pop.where([split])
        if len(ev) == 0:
            continue
        try:
            metrics, _, scored[split] = evaluate(predictor, persistent, ev, best, int(t.n_boot), run.rng)
        except Exception as e:  # e.g. an architecture the backend cannot load yet; the artifact is saved
            log.error("eval %s failed (%s: %s); skipping this split", split, type(e).__name__, e)
            summary[f"eval/{split}/error"] = f"{type(e).__name__}: {e}"
            continue
        summary.update({f"eval/{split}/{k}": v for k, v in metrics.items()})
        log.info("eval %-6s n=%-4d spearman=%.3f rmse=%.2f loss=%.3f", split, len(ev), metrics["spearman/mean"],
                 metrics["rmse/mean"], metrics["loss"])
    if t.get("baselines"):   # prompt-free reference points on the same models
        base = baseline_metrics(list(t.baselines), tr, scored, lambda: build_readout(cfg.readout), grid,
                                bool(cfg.episodes.standardize))
        summary.update(base)
        for name in t.baselines:
            log.info("baseline %-10s train lofo=%.3f  %s", name, base[f"baseline/{name}/train/lofo_loss"],
                     "  ".join(f"{s} loss={base[f'baseline/{name}/{s}/loss']:.3f}" for s in scored))
    (run.out_dir / "metrics.json").write_text(json.dumps(summary, indent=1, default=float))
    run.logger.summary(summary)
    run.logger.log_artifact("onepunch_artifact", "model", [art / f for f in
                            ("prompt.txt", "predictor.npz", "config.yaml", "train_models.txt")])
    run.logger.finish()
    log.info("artifact: %s", art)
    return art


# --------------------------------------------------------------------------- inference

def _resolve_specs(cfg: DictConfig) -> list[ModelSpec]:
    reg = Registry.load(cfg.paths.registry)
    p = cfg.predict
    if p.model_ids:
        known = set(reg.model_ids())
        return [reg.spec(m) if m in known else ModelSpec(model_id=m) for m in p.model_ids]
    pool = select_pool(reg, cfg.pool)
    if p.splits:
        pool = pool.filter(split=list(p.splits))
    return pool.specs()


def predict(cfg: DictConfig) -> pd.DataFrame:
    """Predict all K scores for the requested models; adds ground truth and metrics where known."""
    if not cfg.predict.artifact:
        raise SystemExit("set predict.artifact=<run dir>/artifact")
    run = start_run(cfg, "predict")
    prompt, predictor = load_artifact(cfg.predict.artifact)
    meta = predictor.meta
    extractor = build_feature_source(_feature_cfg_from_meta(cfg, meta))
    specs = _resolve_specs(cfg)
    specs = [dataclasses.replace(s, dtype=meta.get("dtype") or s.dtype) for s in specs]

    Phi = extractor.features(specs, [prompt])[:, 0]
    pred = predictor.predict(Phi)
    benches = meta["benchmarks"]
    df = pd.DataFrame(pred, columns=benches)
    df.insert(0, "model_id", [s.model_id for s in specs])

    S, mask = ScoreTable.load(cfg.paths.scores).matrix(df["model_id"].tolist(), benches,
                                                       sources=tuple(cfg.scores.sources),
                                                       protocol_version=cfg.scores.protocol_version)
    for k, b in enumerate(benches):
        df[f"true/{b}"] = S[:, k]
    full = mask.all(axis=1)
    if full.sum() >= 3:
        fams = np.array([s.family or s.model_id for s in specs])[full]
        metrics = summarize(pred[full], S[full], fams, benches)
        (run.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
        run.logger.summary(metrics)
        log.info("ground truth for %d models: spearman=%.3f rmse=%.2f", int(full.sum()),
                 metrics["spearman/mean"], metrics["rmse/mean"])
    df.to_csv(run.out_dir / "predictions.csv", index=False)
    run.logger.log_table("predict/predictions", list(df.columns), df.itertuples(index=False))
    run.logger.finish()
    log.info("predictions: %s", run.out_dir / "predictions.csv")
    return df
