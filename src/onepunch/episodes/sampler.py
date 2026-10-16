"""Episode sampling: family-disjoint (and optionally tokenizer-disjoint) inner/outer splits."""

from __future__ import annotations

import warnings
from typing import Sequence

import numpy as np

from onepunch.types import Episode


class EpisodeSampler:
    """Draws `n_episodes` episodes over a model population (indices into the given arrays).

    Standard episode: families are split at random into an inner and an outer group; each
    set is filled by picking a family of its group uniformly, then one of its models
    uniformly, without replacement, at most `n_fam_cap` per family.
    With prob `rho` the episode is tokenizer-disjoint instead: out = one held-out tokenizer's
    models, in = models of the other tokenizers. If an active surrogate group is given
    (bool mask), its families go inner in half of the standard episodes and outer in the
    other half, inner in tokenizer-disjoint ones, and >= 1 active model is drawn per episode.
    Infeasible sizes give smaller sets (warned once); disjointness is never violated.
    """

    def __init__(self, n_in: int, n_out: int, n_fam_cap: int, rho: float, n_episodes: int):
        self.n_in, self.n_out, self.n_fam_cap = int(n_in), int(n_out), int(n_fam_cap)
        self.rho, self.n_episodes = float(rho), int(n_episodes)
        self._warned: set[str] = set()

    @classmethod
    def from_cfg(cls, cfg) -> "EpisodeSampler":
        """Build from the `episodes` config group (extra keys are ignored)."""
        return cls(cfg["n_in"], cfg["n_out"], cfg["n_fam_cap"], cfg["rho"], cfg["n_episodes"])

    def _warn(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(f"EpisodeSampler: {msg}", stacklevel=3)

    # ------------------------------------------------------------------ public
    def sample(self, families: Sequence, tokenizers: Sequence, rng: np.random.Generator,
               active_mask: np.ndarray | None = None,
               active_tokenizer: str | None = None) -> list[Episode]:
        fam = np.asarray(families)
        tok = np.asarray(tokenizers)
        n = len(fam)
        active = np.zeros(n, bool) if active_mask is None else np.asarray(active_mask, bool)
        episodes: list[Episode] = []
        std_count = int(rng.integers(2))  # random parity: which half starts with active-inner
        for _ in range(self.n_episodes):
            disjoint = rng.random() < self.rho
            ep = self._tok_disjoint(fam, tok, rng, active, active_tokenizer) if disjoint else None
            if ep is None:
                ep = self._standard(fam, rng, active, active_in=(std_count % 2 == 0))
                std_count += 1
            if len(ep.in_idx) < 2 or len(ep.out_idx) < 1:
                self._warn("empty", "infeasible episode (too few models/families); skipped")
                continue
            if len(ep.in_idx) < self.n_in or len(ep.out_idx) < self.n_out:
                self._warn("sizes", f"requested sizes ({self.n_in}, {self.n_out}) not reachable "
                           f"with {len(set(fam))} families / {n} models and cap {self.n_fam_cap}; "
                           "using smaller sets")
            episodes.append(ep)
        return episodes

    # ---------------------------------------------------------------- internals
    def _fill(self, group: list, by_fam: dict, n: int, rng: np.random.Generator,
              first: int | None = None, fam_of: dict | None = None,
              allowed: np.ndarray | None = None) -> list[int]:
        """Pick up to n models from the families in `group` (family-uniform, capped)."""
        remaining = {}
        for f in group:
            idx = by_fam[f] if allowed is None else by_fam[f][allowed[by_fam[f]]]
            remaining[f] = list(rng.permutation(idx))
        chosen: list[int] = []
        counts = {f: 0 for f in group}
        if first is not None:
            chosen.append(int(first))
            f0 = fam_of[int(first)]
            counts[f0] += 1
            remaining[f0] = [i for i in remaining[f0] if i != first]
        while len(chosen) < n:
            avail = [f for f in group if remaining[f] and counts[f] < self.n_fam_cap]
            if not avail:
                break
            f = avail[int(rng.integers(len(avail)))]
            chosen.append(int(remaining[f].pop()))
            counts[f] += 1
        return chosen

    @staticmethod
    def _groups(fam: np.ndarray) -> tuple[list, dict, dict]:
        fams = sorted(set(fam.tolist()))
        by_fam = {f: np.flatnonzero(fam == f) for f in fams}
        fam_of = {int(i): f for f, idx in by_fam.items() for i in idx}
        return fams, by_fam, fam_of

    def _pick_active(self, cand_fams: list, by_fam: dict, active: np.ndarray,
                     rng: np.random.Generator, allowed: np.ndarray | None = None) -> int | None:
        ok = active if allowed is None else (active & allowed)
        fams = [f for f in cand_fams if ok[by_fam[f]].any()]
        if not fams:
            return None
        f = fams[int(rng.integers(len(fams)))]
        idx = by_fam[f][ok[by_fam[f]]]
        return int(idx[int(rng.integers(len(idx)))])

    def _standard(self, fam: np.ndarray, rng: np.random.Generator, active: np.ndarray,
                  active_in: bool) -> Episode:
        fams, by_fam, fam_of = self._groups(fam)
        F = len(fams)
        act_f = [f for f in fams if active[by_fam[f]].any()]
        other = [fams[i] for i in rng.permutation(F) if fams[i] not in act_f]
        k_in = F if F < 2 else int(np.clip(round(F * self.n_in / (self.n_in + self.n_out)), 1, F - 1))
        if act_f:
            k_side = k_in if active_in else F - k_in
            k_side = min(max(k_side, len(act_f)), F - 1 if other else F)
            side, rest = act_f + other[:k_side - len(act_f)], other[k_side - len(act_f):]
            g_in, g_out = (side, rest) if active_in else (rest, side)
        else:
            g_in, g_out = other[:k_in], other[k_in:]
        if not g_in or not g_out:
            self._warn("groups", "fewer than 2 families: cannot make family-disjoint episodes")
        first = self._pick_active(act_f, by_fam, active, rng) if act_f else None
        in_idx = self._fill(g_in, by_fam, self.n_in, rng, first if active_in else None, fam_of)
        out_idx = self._fill(g_out, by_fam, self.n_out, rng, None if active_in else first, fam_of)
        return Episode(np.sort(np.array(in_idx, dtype=np.int64)),
                       np.sort(np.array(out_idx, dtype=np.int64)), False)

    def _tok_disjoint(self, fam: np.ndarray, tok: np.ndarray, rng: np.random.Generator,
                      active: np.ndarray, active_tokenizer: str | None) -> Episode | None:
        cands = [t for t in sorted(set(tok.tolist())) if t != active_tokenizer and (tok != t).any()]
        if active.any():  # an active model of another tokenizer must remain for the inner set
            cands = [t for t in cands if (active & (tok != t)).any()]
        if not cands:
            self._warn("tokdisjoint", "no valid held-out tokenizer; falling back to standard episode")
            return None
        t = cands[int(rng.integers(len(cands)))]
        fams, by_fam, fam_of = self._groups(fam)
        first = None
        if active.any():
            first = self._pick_active(fams, by_fam, active, rng, allowed=(tok != t))
        out_fams = [f for f in fams if f != (fam_of[first] if first is not None else None)
                    and (tok[by_fam[f]] == t).any()]
        out_idx = self._fill(out_fams, by_fam, self.n_out, rng, allowed=(tok == t))
        used = {fam_of[i] for i in out_idx}
        in_fams = [f for f in fams if f not in used and (tok[by_fam[f]] != t).any()]
        in_idx = self._fill(in_fams, by_fam, self.n_in, rng, first, fam_of, allowed=(tok != t))
        return Episode(np.sort(np.array(in_idx, dtype=np.int64)),
                       np.sort(np.array(out_idx, dtype=np.int64)), True)


def sample_step_pool(families: Sequence, rng: np.random.Generator, n_families: int,
                     models_per_family: int, eligible_mask: np.ndarray | None = None,
                     must_include_mask: np.ndarray | None = None) -> np.ndarray:
    """Per-step model pool: `n_families` random families x `models_per_family` random models.

    `eligible_mask` restricts candidates; `must_include_mask` (e.g. the active surrogate
    group) forces at least one family with such a model, and that model, into the pool.
    Returns sorted indices into `families`.
    """
    fam = np.asarray(families)
    ok = np.ones(len(fam), bool) if eligible_mask is None else np.asarray(eligible_mask, bool)
    fams = sorted(set(fam[ok].tolist()))
    must = None if must_include_mask is None else (np.asarray(must_include_mask, bool) & ok)
    forced_f = None
    forced_i = None
    if must is not None and must.any():
        forced_i = int(rng.choice(np.flatnonzero(must)))
        forced_f = fam[forced_i]
    order = [fams[i] for i in rng.permutation(len(fams))]
    if forced_f is not None:
        order = [forced_f] + [f for f in order if f != forced_f]
    picked: list[int] = []
    for f in order[:n_families]:
        idx = np.flatnonzero((fam == f) & ok)
        take = list(rng.permutation(idx)[:models_per_family])
        if f == forced_f and forced_i not in take:
            take = [forced_i] + take[:models_per_family - 1]
        picked.extend(int(i) for i in take)
    return np.array(sorted(picked), dtype=np.int64)
