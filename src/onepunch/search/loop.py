"""Algorithm 1 control flow: propose -> filter -> exact episodic check -> keep best; validation + early stop."""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from onepunch.search.constraints import PromptConstraints
from onepunch.types import Optimizer, SearchState
from onepunch.utils.wandb_logger import Logger

ActiveSchedule = Callable[[int], "tuple[np.ndarray | None, str | None]"]


log = logging.getLogger(__name__)

@dataclass
class SearchResult:
    best_prompt: bytes
    best_val_loss: float           # NaN if no validation models
    final_prompt: bytes
    history: list[dict[str, Any]] = field(default_factory=list)
    best_step_loss: float = math.nan
    init_val_loss: float = math.nan  # validation loss of the initial prompt (NaN if no validation models)


def _text(p: bytes) -> str:
    return p.decode("utf-8", errors="replace")


class SearchLoop:
    """Drives any Optimizer (`propose(state) -> list[bytes]`) against an objective.

    objective: callable list[bytes] -> losses[P], with `new_step(rng, active_mask, active_tokenizer)`
    (and optionally `extras()`). The current prompt is always scored together with the candidates on
    the same episodes, so the step loss never increases.
    val: callable bytes -> float. The best prompt is chosen by validation loss (checked on the
    initial prompt, every `val_every` steps and at the end); stops after `patience` checks
    without improvement. Without `val`, the best prompt is the one with the lowest step loss.
    Optional hooks: `optimizer.before_step(objective, state)` (e.g. GCG computes gradients there);
    `active_schedule(step) -> (active_mask, active_tokenizer)` rotates surrogate groups.
    """

    def __init__(self, optimizer: Optimizer, objective: Any, val: Callable[[bytes], float] | None = None,
                 constraints: PromptConstraints | None = None, steps: int = 100, val_every: int = 10,
                 patience: int = 5, logger: Logger | None = None, log_prompt_every: int = 10,
                 active_schedule: ActiveSchedule | None = None):
        self.optimizer, self.objective, self.val, self.constraints = optimizer, objective, val, constraints
        self.steps, self.val_every, self.patience = int(steps), int(val_every), int(patience)
        self.logger = logger if logger is not None else Logger({"mode": "disabled"})
        self.log_prompt_every = int(log_prompt_every)
        self.active_schedule = active_schedule

    def run(self, init_prompt: bytes, rng: np.random.Generator) -> SearchResult:
        t0 = time.time()
        current, cur_loss = init_prompt, math.nan
        history: list[dict[str, Any]] = []
        best, best_val, best_step_loss = init_prompt, math.inf, math.inf
        stall, table_rows, init_val = 0, [], math.nan
        if self.val is not None:
            best_val = init_val = float(self.val(current))
            log.info("step 0/%d (initial prompt) val=%.4f %.0fs", self.steps, init_val, time.time() - t0)
            self.logger.log({"val/loss": best_val, "val/best_loss": best_val, "val/stall": 0,
                             "search/prompt_bytes": len(current)}, step=0)
        last_validated = True

        for t in range(self.steps):
            mask, active_tok = self.active_schedule(t) if self.active_schedule else (None, None)
            self.objective.new_step(rng, mask, active_tok)
            state = SearchState(step=t, prompt=current, loss=cur_loss, best_prompt=best,
                                best_loss=best_val if self.val else best_step_loss, rng=rng,
                                history=history)
            if hasattr(self.objective, "extras"):
                state.extras.update(self.objective.extras())
            if hasattr(self.optimizer, "before_step"):
                self.optimizer.before_step(self.objective, state)

            cands = [c for c in dict.fromkeys(self.optimizer.propose(state)) if c != current]
            n_prop = len(cands)
            if self.constraints is not None:
                cands = self.constraints.filter(cands)
            losses = np.asarray(self.objective([current] + cands), dtype=np.float64)
            j = int(np.argmin(losses))  # ties keep the current prompt (index 0)
            current, cur_loss = ([current] + cands)[j], float(losses[j])
            rec: dict[str, Any] = {"step": t, "loss": cur_loss, "loss_before": float(losses[0]),
                                   "accepted": j > 0,
                                   "n_proposed": n_prop, "n_valid": len(cands),
                                   "wall_time": time.time() - t0}
            if self.val is None and cur_loss < best_step_loss:
                best, best_step_loss = current, cur_loss
            last_validated = False
            stop = False
            if self.val is not None and (t + 1) % self.val_every == 0:
                m = float(self.val(current))
                rec["val_loss"], last_validated = m, True
                if m < best_val:
                    best, best_val, stall = current, m, 0
                else:
                    stall += 1
                    stop = stall >= self.patience
            history.append(rec)
            log.info("step %d/%d loss=%.4f (before %.4f) valid=%d/%d%s %.0fs", t + 1, self.steps, cur_loss,
                     rec.get("loss_before", math.nan), len(cands), rec.get("n_proposed", len(cands)),
                     f" val={rec['val_loss']:.4f}" if "val_loss" in rec else "", rec["wall_time"])
            # train/loss_before: the kept prompt on this step's fresh episodes (unbiased);
            # train/loss: after picking the best candidate on those same episodes (optimistic).
            metrics = {"train/loss": cur_loss, "train/loss_before": rec["loss_before"],
                       "train/gain": rec["loss_before"] - cur_loss, "search/accepted": int(j > 0),
                       "search/n_proposed": n_prop, "search/n_valid": len(cands),
                       "search/prompt_bytes": len(current), "search/wall_time": rec["wall_time"]}
            if "val_loss" in rec:
                metrics.update({"val/loss": rec["val_loss"], "val/best_loss": best_val,
                                "val/stall": stall})
            self.logger.log(metrics, step=t + 1)
            if self.log_prompt_every and (t + 1) % self.log_prompt_every == 0:
                table_rows.append((t + 1, cur_loss, _text(current)))
                self.logger.log_table("search/prompts", ["step", "loss", "prompt"], table_rows, step=t + 1)
            if stop:
                break

        if self.val is not None and not last_validated:
            m = float(self.val(current))
            if history:
                history[-1]["val_loss"] = m
            if m < best_val:
                best, best_val = current, m
            self.logger.log({"val/loss": m, "val/best_loss": best_val}, step=len(history))
        return SearchResult(best_prompt=best, best_val_loss=best_val if self.val else math.nan,
                            final_prompt=current, history=history,
                            best_step_loss=best_step_loss if self.val is None else math.nan,
                            init_val_loss=init_val)
