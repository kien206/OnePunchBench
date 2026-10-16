"""Search objectives: the episodic step loss L(p) and the monitor loss."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch

from onepunch.episodes.loss import episode_loss, monitor_loss
from onepunch.episodes.sampler import EpisodeSampler, sample_step_pool
from onepunch.readout.base import Readout
from onepunch.types import Episode, FeatureSource, ModelSpec


class EpisodicObjective:
    """L(p) = mean over the step's episodes of the outer loss, for many prompts at once.

    Call `new_step(rng)` once per step: it (maybe) redraws the step model pool and draws fresh
    episodes. `__call__(prompts)` then fetches features of every step model for every prompt in ONE
    FeatureSource call and returns the episode-averaged loss per prompt (no grad). All candidates of
    a step are therefore scored on the same episodes.

    models/S/families/tokenizers are aligned arrays over the training models. `step_pool` is
    None (use all models) or {"n_families", "models_per_family", "redraw_every"}; episodes are drawn
    inside the pool.
    """

    def __init__(self, source: FeatureSource, models: Sequence[ModelSpec], S: np.ndarray,
                 families: Sequence, tokenizers: Sequence, sampler: EpisodeSampler, readout: Readout,
                 sigma: np.ndarray | None, step_pool: dict | None = None, center: bool = True):
        self.source, self.models = source, list(models)
        self.S = np.asarray(S, dtype=np.float64)
        self.families, self.tokenizers = np.asarray(families), np.asarray(tokenizers)
        self.sampler, self.readout, self.sigma = sampler, readout, sigma
        self.step_pool, self.center = step_pool, center
        self.n_steps = 0
        self._pool: np.ndarray | None = None
        self._pool_key: Any = None
        self.rows = np.arange(0)           # global indices of the models used this step
        self.episodes: list[Episode] = []  # indices into `rows`

    def new_step(self, rng: np.random.Generator, active_mask: np.ndarray | None = None,
                 active_tokenizer: str | None = None) -> None:
        N = len(self.models)
        act = None if active_mask is None else np.asarray(active_mask, bool)
        key = None if act is None else act.tobytes()
        if self.step_pool is None:
            pool = np.arange(N)
        else:
            every = int(self.step_pool.get("redraw_every", 10))
            if self._pool is None or self.n_steps % every == 0 or key != self._pool_key:
                self._pool = sample_step_pool(self.families, rng, self.step_pool["n_families"],
                                              self.step_pool["models_per_family"], must_include_mask=act)
                self._pool_key = key
            pool = self._pool
        eps = self.sampler.sample(self.families[pool], self.tokenizers[pool], rng,
                                  None if act is None else act[pool], active_tokenizer)
        if not eps:
            raise RuntimeError("no feasible episodes for this step's model pool")
        used = np.unique(np.concatenate([np.concatenate([e.in_idx, e.out_idx]) for e in eps]))
        remap = np.full(len(pool), -1)
        remap[used] = np.arange(len(used))
        self.rows = pool[used]
        self.episodes = [Episode(remap[e.in_idx], remap[e.out_idx], e.tokenizer_disjoint) for e in eps]
        self.n_steps += 1

    @property
    def step_models(self) -> list[ModelSpec]:
        return [self.models[i] for i in self.rows]

    def extras(self) -> dict[str, Any]:
        """What gradient-based optimizers need (for SearchState.extras)."""
        return {"episodes": self.episodes, "step_models": self.step_models, "step_rows": self.rows,
                "S_step": self.S[self.rows], "sigma": self.sigma, "readout": self.readout}

    def features(self, prompts: Sequence[bytes]) -> np.ndarray:
        """[N_step, P, d] features of the step's models."""
        return self.source.features(self.step_models, list(prompts))

    def __call__(self, prompts: Sequence[bytes]) -> np.ndarray:
        if not self.episodes:
            raise RuntimeError("call new_step() before scoring")
        Phi = torch.as_tensor(self.features(prompts), dtype=torch.float64)
        with torch.no_grad():
            loss = episode_loss(Phi, self.S[self.rows], self.episodes, self.readout, self.sigma, self.center)
        return loss.numpy()


class MonitorObjective:
    """Monitor loss: readout fitted on all training models, error on the monitor models."""

    def __init__(self, source: FeatureSource, train_models: Sequence[ModelSpec], S_train: np.ndarray,
                 mon_models: Sequence[ModelSpec], S_mon: np.ndarray, readout: Readout,
                 sigma: np.ndarray | None, center: bool = True):
        self.source, self.readout, self.sigma, self.center = source, readout, sigma, center
        self.train_models, self.mon_models = list(train_models), list(mon_models)
        self.S_train, self.S_mon = np.asarray(S_train, dtype=np.float64), np.asarray(S_mon, dtype=np.float64)

    def __call__(self, prompt: bytes) -> float:
        Phi = self.source.features(self.train_models + self.mon_models, [prompt])[:, 0, :]
        n = len(self.train_models)
        with torch.no_grad():
            return float(monitor_loss(Phi[:n], self.S_train, Phi[n:], self.S_mon, self.readout,
                                      self.sigma, self.center))
