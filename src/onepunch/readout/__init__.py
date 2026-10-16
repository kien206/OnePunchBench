from __future__ import annotations

from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from onepunch.readout.base import FittedReadout, Readout
from onepunch.readout.ridge import FittedRidge, RidgeReadout

__all__ = ["Readout", "FittedReadout", "RidgeReadout", "FittedRidge", "build_readout",
           "lambda0_grid"]


def build_readout(cfg: DictConfig | dict) -> Readout:
    """Instantiate `cfg.readout`; the `cv:` block (lambda0 grid etc.) is not a constructor arg."""
    c = OmegaConf.to_container(OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg, resolve=True)
    c.pop("cv", None)
    return instantiate(c)


def lambda0_grid(cfg: DictConfig | dict) -> list[float]:
    """The λ0 grid for family-grouped CV from the readout config."""
    c: Any = OmegaConf.to_container(OmegaConf.create(cfg) if isinstance(cfg, dict) else cfg, resolve=True)
    return [float(x) for x in c["cv"]["lambda0_grid"]]
