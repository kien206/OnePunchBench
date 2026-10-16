"""Thin wandb wrapper. `mode: disabled` makes every call a no-op, so library
code and tests never need to check whether logging is on."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

from omegaconf import DictConfig, OmegaConf


class Logger:
    def __init__(self, cfg: DictConfig | dict, run_cfg: DictConfig | dict | None = None,
                 name: str | None = None):
        """cfg: the `logger` config group; run_cfg: full resolved config to store with the run."""
        cfg = OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg
        self.mode = cfg.get("mode", "online")
        self._run = None
        if self.mode == "disabled":
            return
        import wandb

        config = None
        if run_cfg is not None:
            config = OmegaConf.to_container(OmegaConf.create(run_cfg), resolve=True)
        self._run = wandb.init(
            project=cfg.get("project", "onepunchbench"),
            entity=cfg.get("entity"),
            group=cfg.get("group"),
            tags=list(cfg.get("tags") or []),
            name=name or cfg.get("name"),
            mode=self.mode,
            dir=cfg.get("dir"),
            config=config,
        )

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if self._run is not None:
            self._run.log(metrics, step=step)

    def log_table(self, key: str, columns: Sequence[str], rows: Iterable[Sequence[Any]],
                  step: int | None = None) -> None:
        """E.g. per-step prompts: columns=["step", "loss", "prompt"]."""
        if self._run is None:
            return
        import wandb

        self._run.log({key: wandb.Table(columns=list(columns), data=[list(r) for r in rows])},
                      step=step)

    def summary(self, metrics: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.summary.update(metrics)

    def log_artifact(self, name: str, type: str, paths: Sequence[str | Path],
                     metadata: dict[str, Any] | None = None) -> None:
        """Store files (e.g. p*, W*, feature/score means) as a versioned artifact."""
        if self._run is None:
            return
        import wandb

        art = wandb.Artifact(name=name, type=type, metadata=metadata)
        for p in paths:
            art.add_file(str(p))
        self._run.log_artifact(art)

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None
