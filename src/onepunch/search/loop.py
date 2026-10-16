"""Algorithm 1 control flow: propose -> filter -> exact episodic check -> keep best; monitor + early stop."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from onepunch.search.constraints import PromptConstraints
from onepunch.types import Optimizer, SearchState
from onepunch.utils.wandb_logger import Logger

ActiveSchedule = Callable[[int], "tuple[np.ndarray | None, str | None]"]


@dataclass
class SearchResult:
    best_prompt: bytes
    best_monitor_loss: float       # NaN if no monitor
    final_prompt: bytes
    history: list[dict[str, Any]] = field(default_factory=list)
    best_step_loss: float = math.nan


def _text(p: bytes) -> str:
    return p.decode("utf-8", errors="replace")


class SearchLoop:
    """Drives any Optimizer (`propose(state) -> list[bytes]`) against an objective.

    objective: callable list[bytes] -> losses[P], with `new_step(rng, active_mask, active_tokenizer)`
    (and optionally `extras()`). The current prompt is always scored together with the candidates on
    the same episodes, so the step loss never increases.
    monitor: callable bytes -> float. The best prompt is chosen by monitor loss (checked on the
    initial prompt, every `monitor_every` steps and at the end); stops after `patience` checks
    without improvement. Without a monitor, the best prompt is the one with the lowest step loss.
    Optional hooks: `optimizer.before_step(objective, state)` (e.g. GCG computes gradients there);
    `active_schedule(step) -> (active_mask, active_tokenizer)` rotates surrogate groups.
    """

    def __init__(self, optimizer: Optimizer, objective: Any, monitor: Callable[[bytes], float] | None = None,
                 constraints: PromptConstraints | None = None, steps: int = 100, monitor_every: int = 10,
                 patience: int = 5, logger: Logger | None = None, log_prompt_every: int = 10,
                 active_schedule: ActiveSchedule | None = None):
        self.optimizer, self.objective, self.monitor, self.constraints = optimizer, objective, monitor, constraints
        self.steps, self.monitor_every, self.patience = int(steps), int(monitor_every), int(patience)
        self.logger = logger if logger is not None else Logger({"mode": "disabled"})
        self.log_prompt_every = int(log_prompt_every)
        self.active_schedule = active_schedule

    def run(self, init_prompt: bytes, rng: np.random.Generator) -> SearchResult:
        t0 = time.time()
        current, cur_loss = init_prompt, math.nan
        history: list[dict[str, Any]] = []
        best, best_mon, best_step_loss = init_prompt, math.inf, math.inf
        stall, table_rows = 0, []
        if self.monitor is not None:
            best_mon = float(self.monitor(current))
            self.logger.log({"search/monitor_loss": best_mon}, step=0)
        last_monitored = True

        for t in range(self.steps):
            mask, active_tok = self.active_schedule(t) if self.active_schedule else (None, None)
            self.objective.new_step(rng, mask, active_tok)
            state = SearchState(step=t, prompt=current, loss=cur_loss, best_prompt=best,
                                best_loss=best_mon if self.monitor else best_step_loss, rng=rng,
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
                                   "n_proposed": n_prop, "n_valid": len(cands),
                                   "wall_time": time.time() - t0}
            if self.monitor is None and cur_loss < best_step_loss:
                best, best_step_loss = current, cur_loss
            last_monitored = False
            stop = False
            if self.monitor is not None and (t + 1) % self.monitor_every == 0:
                m = float(self.monitor(current))
                rec["monitor_loss"], last_monitored = m, True
                if m < best_mon:
                    best, best_mon, stall = current, m, 0
                else:
                    stall += 1
                    stop = stall >= self.patience
            history.append(rec)
            self.logger.log({"search/loss": cur_loss, "search/n_valid": len(cands),
                             "search/wall_time": rec["wall_time"],
                             **({"search/monitor_loss": rec["monitor_loss"]} if "monitor_loss" in rec else {})},
                            step=t + 1)
            if self.log_prompt_every and (t + 1) % self.log_prompt_every == 0:
                table_rows.append((t + 1, cur_loss, _text(current)))
                self.logger.log_table("search/prompts", ["step", "loss", "prompt"], table_rows, step=t + 1)
            if stop:
                break

        if self.monitor is not None and not last_monitored:
            m = float(self.monitor(current))
            if history:
                history[-1]["monitor_loss"] = m
            if m < best_mon:
                best, best_mon = current, m
        return SearchResult(best_prompt=best, best_monitor_loss=best_mon if self.monitor else math.nan,
                            final_prompt=current, history=history,
                            best_step_loss=best_step_loss if self.monitor is None else math.nan)
