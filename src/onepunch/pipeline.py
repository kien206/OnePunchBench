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
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from onepunch.data.registry import Registry, select_pool
from onepunch.data.scores import ScoreTable
from onepunch.episodes import EpisodeSampler, build_step_pool_cfg, select_lambda0, standardize_scales
from onepunch.eval.bootstrap import family_bootstrap
from onepunch.eval.lofo import FittedPredictor, fit_predictor
from onepunch.eval.metrics import mean_spearman, summarize
from onepunch.features import FeatureExtractor
from onepunch.features.hf_backend import load_tokenizer
from onepunch.readout import build_readout, lambda0_grid
from onepunch.search import (EpisodicObjective, MonitorObjective, SearchLoop, build_constraints,
                             build_optimizer)
from onepunch.types import ModelSpec
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
    cfg.features.d = int(meta["d"])
    cfg.features.prefix = meta["prefix"]
    cfg.features.l_chunk = int(meta.get("l_chunk", cfg.features.l_chunk))
    cfg.backend.dtype = meta.get("dtype", cfg.backend.dtype)
    return cfg


# --------------------------------------------------------------------------- evaluation

def evaluate(predictor: FittedPredictor, extractor: FeatureExtractor, pop: Population, prompt: bytes,
             n_boot: int, rng: np.random.Generator) -> tuple[dict[str, float], np.ndarray]:
    """Metrics of the predictor on `pop` (which has ground truth); returns (metrics, predictions)."""
    Phi = extractor.features(pop.specs, [prompt])[:, 0]
    pred = predictor.predict(Phi)
    metrics = summarize(pred, pop.S, pop.families, pop.benchmarks)
    if len(set(pop.families)) > 1:
        _, lo, hi = family_bootstrap(mean_spearman, pred, pop.S, pop.families, n_boot=n_boot, rng=rng)
        metrics.update({"spearman/mean_ci_lo": lo, "spearman/mean_ci_hi": hi})
    metrics["n_models"], metrics["n_families"] = len(pop), len(set(pop.families))
    return metrics, pred


# --------------------------------------------------------------------------- training

def _cv_lambda0(readout, extractor: FeatureExtractor, pop: Population, prompt: bytes, grid) -> None:
    if not hasattr(readout, "lambda0"):
        return
    Phi = extractor.features(pop.specs, [prompt])[:, 0]
    readout.lambda0, table = select_lambda0(Phi, pop.S, pop.families, grid)
    log.info("lambda0 = %g (CV mse: %s)", readout.lambda0, {k: round(v, 4) for k, v in table.items()})


def train(cfg: DictConfig) -> Path:
    """Search p* and fit W*; returns the artifact directory."""
    run = start_run(cfg, "train")
    t, roles = cfg.train, cfg.roles
    pop = load_population(cfg)
    tr, mon = pop.where(roles.train), pop.where(roles.monitor or [])
    if len(set(tr.families)) < 2:
        raise SystemExit(f"need >= 2 training families, got {len(set(tr.families))} (pool={cfg.pool})")
    log.info("train: %d models / %d families; monitor: %d models", len(tr), len(set(tr.families)), len(mon))

    init = load_prompt(cfg, t.init_prompt)
    persistent = FeatureExtractor.from_cfg(cfg)               # named prompts -> disk cache
    source = FeatureExtractor.from_cfg(cfg, persist=False)    # search candidates -> memory only
    readout, grid = build_readout(cfg.readout), lambda0_grid(cfg.readout)
    if t.cv_lambda0:
        _cv_lambda0(readout, persistent, tr, init, grid)
    sigma = standardize_scales(tr.S, bool(cfg.episodes.standardize))

    optimizer = build_optimizer(cfg.search)
    history: list[dict] = []
    if optimizer is None:
        log.info("search=none: fixed-prompt training (readout only)")
        best = init
    else:
        objective = EpisodicObjective(source, tr.specs, tr.S, tr.families, tr.tokenizers,
                                      EpisodeSampler.from_cfg(cfg.episodes), readout, sigma,
                                      step_pool=build_step_pool_cfg(cfg.episodes))
        monitor = MonitorObjective(source, tr.specs, tr.S, mon.specs, mon.S, readout, sigma) if len(mon) else None
        if monitor is None:
            log.warning("no monitor models: the best prompt is chosen by step loss")
        reps = {tok: s for tok, s in zip(pop.tokenizers, pop.specs)}   # one model per tokenizer
        constraints = build_constraints(cfg.search, int(cfg.features.d) * int(cfg.features.l_chunk),
                                        [load_tokenizer(s) for s in reps.values()])
        loop = SearchLoop(optimizer, objective, monitor, constraints, logger=run.logger,
                          **OmegaConf.to_container(cfg.search.loop, resolve=True))
        result = loop.run(init, run.rng)
        best, history = result.best_prompt, result.history
        if t.cv_lambda0:   # formulation: lambda0 re-selected for p*
            _cv_lambda0(readout, persistent, tr, best, grid)

    Phi = persistent.features(tr.specs, [best])[:, 0]
    meta = {"d": int(cfg.features.d), "prefix": str(cfg.features.prefix), "l_chunk": int(cfg.features.l_chunk),
            "dtype": cfg.backend.dtype, "benchmarks": tr.benchmarks, "readout": cfg.readout.get("_target_"),
            "lambda0": getattr(readout, "lambda0", None), "score_sources": list(cfg.scores.sources)}
    predictor = fit_predictor(Phi, tr.S, readout, bool(cfg.episodes.standardize), meta=meta)
    art = save_artifact(run.out_dir / "artifact", best, predictor, cfg, tr.model_ids)
    (run.out_dir / "history.json").write_text(json.dumps(history, indent=1, default=float))

    summary: dict[str, Any] = {"prompt_changed": best != init, "steps_run": len(history)}
    for split in roles.eval or []:
        ev = pop.where([split])
        if len(ev) == 0:
            continue
        try:
            metrics, _ = evaluate(predictor, persistent, ev, best, int(t.n_boot), run.rng)
        except Exception as e:  # e.g. an architecture the backend cannot load yet; the artifact is saved
            log.error("eval %s failed (%s: %s); skipping this split", split, type(e).__name__, e)
            summary[f"eval/{split}/error"] = f"{type(e).__name__}: {e}"
            continue
        summary.update({f"eval/{split}/{k}": v for k, v in metrics.items()})
        log.info("eval %-6s n=%-4d spearman=%.3f rmse=%.2f", split, len(ev), metrics["spearman/mean"],
                 metrics["rmse/mean"])
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
    extractor = FeatureExtractor.from_cfg(_feature_cfg_from_meta(cfg, meta))
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
