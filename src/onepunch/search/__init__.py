from __future__ import annotations

from typing import Any, Sequence

from hydra.utils import instantiate
from omegaconf import OmegaConf

from onepunch.search.best_of_n import BestOfN
from onepunch.search.constraints import PromptConstraints, roundtrip_ok
from onepunch.search.loop import SearchLoop, SearchResult
from onepunch.search.objective import EpisodicObjective, ValidationObjective
from onepunch.search.random_swap import RandomSwap

__all__ = ["SearchLoop", "SearchResult", "PromptConstraints", "roundtrip_ok", "RandomSwap", "BestOfN",
           "EpisodicObjective", "ValidationObjective", "build_optimizer", "build_constraints"]


def build_optimizer(cfg) -> Any | None:
    """Instantiate `cfg.search.optimizer` (None for search=none)."""
    return None if cfg.get("optimizer") is None else instantiate(cfg["optimizer"])


def build_constraints(cfg, min_bytes: int, tokenizers: Sequence[Any] = ()) -> PromptConstraints:
    """From `cfg.search.constraints`; min_bytes = d * L_chunk and the tokenizers come from the caller."""
    c = OmegaConf.to_container(cfg["constraints"], resolve=True) if not isinstance(cfg["constraints"], dict) \
        else dict(cfg["constraints"])
    return PromptConstraints(min_bytes=min_bytes, max_bytes=c["max_bytes"], max_tokens=c.get("max_tokens"),
                             tokenizers=tokenizers)
