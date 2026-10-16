"""Glue shared by experiments/*.py: load the model population, prompts and a run context."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from omegaconf import DictConfig, OmegaConf

from onepunch.data.benchmarks import load_benchmarks
from onepunch.data.registry import Registry, select_pool
from onepunch.data.scores import ScoreTable
from onepunch.types import ModelSpec
from onepunch.utils.seed import seed_everything
from onepunch.utils.wandb_logger import Logger

log = logging.getLogger(__name__)


@dataclass
class Population:
    """Models of one pool with aligned metadata and complete score rows."""

    specs: list[ModelSpec]
    families: np.ndarray     # [N] str
    tokenizers: np.ndarray   # [N] str (unknown tokenizers get a unique placeholder)
    splits: np.ndarray       # [N] str
    params_b: np.ndarray     # [N] float
    S: np.ndarray            # [N, K] raw scores in [0, 1]
    benchmarks: list[str]
    n_items: np.ndarray      # [K]

    def __len__(self) -> int:
        return len(self.specs)

    @property
    def model_ids(self) -> list[str]:
        return [s.model_id for s in self.specs]

    def subset(self, mask: np.ndarray) -> "Population":
        idx = np.flatnonzero(mask)
        return Population([self.specs[i] for i in idx], self.families[idx], self.tokenizers[idx],
                          self.splits[idx], self.params_b[idx], self.S[idx], self.benchmarks, self.n_items)

    def where(self, splits: Sequence[str]) -> "Population":
        return self.subset(np.isin(self.splits, list(splits)))


def load_population(cfg: DictConfig, splits: Sequence[str] | None = None) -> Population:
    """Pool from `cfg.pool`, optionally restricted to `splits`, keeping models with all K scores."""
    reg = select_pool(Registry.load(cfg.paths.registry), cfg.pool)
    if splits:
        reg = reg.filter(split=list(splits))
    benches = load_benchmarks(cfg.paths.benchmarks)
    sc = cfg.get("scores") or {}
    names = list(sc.get("benchmarks") or benches)
    missing = [b for b in names if b not in benches]
    if missing:
        raise ValueError(f"scores.benchmarks {missing} not in {cfg.paths.benchmarks}")
    S, mask = ScoreTable.load(cfg.paths.scores).matrix(
        reg.model_ids(), names, sources=tuple(sc.get("sources", ["archive"])),
        protocol_version=sc.get("protocol_version", "oll_v2"))
    keep = mask.all(axis=1)
    if (~keep).any():
        log.warning("dropping %d models with missing scores", int((~keep).sum()))
    df = reg.df.reset_index(drop=True)
    tok = [t if isinstance(t, str) and t else f"unknown:{m}" for t, m in zip(df["tokenizer_hash"], df["model_id"])]
    pop = Population(
        specs=reg.specs(), families=df["family"].astype(str).to_numpy(), tokenizers=np.asarray(tok),
        splits=df["split"].astype(str).to_numpy(), params_b=df["params_b"].astype(float).to_numpy(),
        S=S, benchmarks=names, n_items=np.array([benches[b].n_items for b in names], dtype=float),
    )
    return pop.subset(keep)


def load_prompt(cfg: DictConfig, name: str) -> bytes:
    """Read data/prompts/<name>.txt as raw bytes."""
    return (Path(cfg.paths.prompts) / f"{name}.txt").read_bytes()


@dataclass
class RunContext:
    cfg: DictConfig
    rng: np.random.Generator
    logger: Logger
    out_dir: Path
    exp: dict[str, Any] = field(default_factory=dict)


def start_run(cfg: DictConfig, name: str, exp_defaults: dict[str, Any] | None = None) -> RunContext:
    """Seed, open the logger (group defaults to the experiment name) and merge experiment options.

    Experiment-specific options live in the script as `exp_defaults` and are overridden from the
    CLI with `+exp.<key>=<value>`.
    """
    exp = OmegaConf.merge(OmegaConf.create(exp_defaults or {}), cfg.get("exp") or {})
    OmegaConf.set_struct(cfg, False)
    cfg.exp = exp
    if cfg.logger.get("group") is None:
        cfg.logger.group = name
    try:
        from hydra.core.hydra_config import HydraConfig

        out_dir = Path(HydraConfig.get().runtime.output_dir)
    except Exception:  # not launched through @hydra.main
        out_dir = Path(cfg.paths.runs) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = seed_everything(int(cfg.seed))
    logger = Logger(cfg.logger, run_cfg=cfg, name=cfg.logger.get("name"))
    return RunContext(cfg, rng, logger, out_dir, OmegaConf.to_container(exp, resolve=True))
