"""E0 feature checks (initial_plan.md, E0 row): are the chunk features deterministic enough?

For every pool model and dtype, the generic passage is scored four ways:
  run A (batch 1), run B (batch 1 again), run C (batch 4, mixed with other prompts),
and in each dtype we report
  - repeat noise  |A - B|, batch noise |A - C| (per chunk feature),
  - the between-model spread (std over models of each chunk feature),
  - noise / spread (pass if < exp.max_noise_ratio, default 1%),
  - the chunk-sum identity  sum_j |c_j| f_j == sum_u logprob_u,
plus bf16-vs-fp32 differences.

  python experiments/e0_feature_checks.py pool=smoke logger.mode=offline
"""

from __future__ import annotations

import dataclasses
import json
import logging

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from onepunch.features import HFBackend, chunk_features
from onepunch.features.chunking import chunk_sizes
from onepunch.search import BestOfN
from onepunch.types import SearchState
from onepunch.utils.experiment import load_population, load_prompt, start_run

log = logging.getLogger(__name__)

EXP_DEFAULTS = {
    "splits": None,
    "dtypes": ["bfloat16", "float32"],
    "max_noise_ratio": 0.01,
    "batch_size": 4,
}


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run = start_run(cfg, "e0_feature_checks", EXP_DEFAULTS)
    exp, d = run.exp, int(cfg.features.d)
    prefix = str(cfg.features.prefix).encode()
    pop = load_population(cfg, exp["splits"])
    p = load_prompt(cfg, "generic_passage")
    state = SearchState(step=0, prompt=b"", loss=np.nan, best_prompt=b"", best_loss=np.nan, rng=run.rng)
    fillers = [load_prompt(cfg, "handwritten_probe")] + BestOfN("random_ascii", 1500, 2).propose(state)
    sizes = chunk_sizes(len(p), d)

    feats: dict[tuple[str, str], np.ndarray] = {}   # (dtype, run) -> [N, d]
    sum_err = []
    for dt in exp["dtypes"]:
        A, B, C = [], [], []
        for spec in pop.specs:
            spec = dataclasses.replace(spec, dtype=dt)
            b1 = HFBackend(spec, device=cfg.device, batch_size=1,
                           attn_implementation=cfg.backend.attn_implementation)
            ta = b1.token_logprobs([p], prefix)[0]
            tb = b1.token_logprobs([p], prefix)[0]
            b4 = HFBackend(spec, device=cfg.device, batch_size=exp["batch_size"], model=b1.model,
                           tokenizer=b1.tokenizer, attn_implementation=cfg.backend.attn_implementation)
            tc = b4.token_logprobs([p] + fillers, prefix)[0]
            b1.close()
            del b4
            torch.cuda.empty_cache()
            fa, fb, fc = (chunk_features(t, d) for t in (ta, tb, tc))
            A.append(fa), B.append(fb), C.append(fc)
            sum_err.append(abs(float((sizes * fa).sum()) - float(ta.logprobs.astype(np.float64).sum())))
            log.info("%s [%s] |A-B|max=%.2e |A-C|max=%.2e", spec.model_id, dt,
                     np.abs(fa - fb).max(), np.abs(fa - fc).max())
        for name, arr in zip("ABC", (A, B, C)):
            feats[(dt, name)] = np.stack(arr)

    rows = []
    for dt in exp["dtypes"]:
        A, B, C = feats[(dt, "A")], feats[(dt, "B")], feats[(dt, "C")]
        spread = A.std(axis=0).mean()
        for kind, other in (("repeat", B), ("batch", C)):
            diff = np.abs(A - other)
            rows.append({"dtype": dt, "check": kind, "mean_abs": diff.mean(), "max_abs": diff.max(),
                         "spread": spread, "noise_ratio": diff.mean() / spread,
                         "pass": bool(diff.mean() / spread < exp["max_noise_ratio"])})
    if {"bfloat16", "float32"} <= set(exp["dtypes"]):
        diff = np.abs(feats[("bfloat16", "A")] - feats[("float32", "A")])
        spread = feats[("float32", "A")].std(axis=0).mean()
        rows.append({"dtype": "bf16_vs_fp32", "check": "dtype", "mean_abs": diff.mean(), "max_abs": diff.max(),
                     "spread": spread, "noise_ratio": diff.mean() / spread,
                     "pass": bool(diff.mean() / spread < exp["max_noise_ratio"])})
    table = pd.DataFrame(rows)
    summary = {"chunk_sum_max_abs_err": max(sum_err), "n_models": len(pop),
               "all_pass": bool(table["pass"].all())}

    table.to_csv(run.out_dir / "e0_feature_checks.csv", index=False)
    (run.out_dir / "e0_summary.json").write_text(json.dumps(summary, indent=1))
    np.savez(run.out_dir / "e0_features.npz", model_ids=np.array(pop.model_ids),
             **{f"{dt}_{r}": v for (dt, r), v in feats.items()})
    run.logger.log_table("e0/checks", list(table.columns), table.itertuples(index=False))
    run.logger.summary(summary)
    run.logger.finish()
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
