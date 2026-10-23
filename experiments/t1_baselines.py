"""T1: does a fixed prompt say more than model size? (initial_plan.md, Gate 1)

Leave-one-family-out ridge on fixed-prompt baselines, scored with the same readout:
  1. log parameter count alone            4. random bytes / random tokens (d)
  2. generic passage at d=1 (bits/byte)   5. hand-written probe (d)
  3. generic passage at d                 6. generic passage (d) + log params
lambda0 is chosen by family-grouped CV inside each LOFO fold (nested, no leakage).

  python experiments/t1_baselines.py                     # smoke pool, d=16
  python experiments/t1_baselines.py pool=tier_s +exp.splits=[train,val] logger.mode=offline
"""

from __future__ import annotations

import json
import logging

import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig

from onepunch.data.benchmarks import noise_ceiling_rmse
from onepunch.episodes import fit_predict, select_lambda0, standardize_scales
from onepunch.eval.bootstrap import family_bootstrap
from onepunch.eval.metrics import mean_rmse, mean_spearman, mean_within_spearman, summarize
from onepunch.features import FeatureExtractor, chunk_features
from onepunch.readout import RidgeReadout, lambda0_grid
from onepunch.search import BestOfN
from onepunch.types import SearchState
from onepunch.utils.experiment import load_population, load_prompt, start_run

log = logging.getLogger(__name__)

EXP_DEFAULTS = {
    "splits": None,                  # e.g. [train, val]; None = every model of the pool
    "random_length_bytes": 2048,
    "random_tokenizer": "Qwen/Qwen2.5-0.5B",
    "n_boot": 1000,
    "gate_margin": 0.05,             # Gate 1: generic passage beats log-params by this mean Spearman
}


def nested_lofo(Phi: np.ndarray, S: np.ndarray, families: np.ndarray, grid: list[float]) -> tuple[np.ndarray, list[float]]:
    """LOFO predictions in raw score space; lambda0 re-selected on each fold's training families."""
    preds, lams = np.empty_like(S), []
    for f in sorted(set(families.tolist())):
        te, tr = families == f, families != f
        lam, _ = select_lambda0(Phi[tr], S[tr], families[tr], grid)
        lams.append(lam)
        sigma = standardize_scales(S[tr])
        with torch.no_grad():
            p = fit_predict(torch.as_tensor(Phi[tr]), torch.as_tensor(S[tr]), torch.as_tensor(Phi[te]),
                            RidgeReadout(lam), sigma)
        preds[te] = np.clip(p.numpy(), 0.0, 1.0)
    return preds, lams


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / (x.std(0) + 1e-12)


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    run = start_run(cfg, "t1_baselines", EXP_DEFAULTS)
    exp, d = run.exp, int(cfg.features.d)
    pop = load_population(cfg, exp["splits"])
    log.info("population: %d models, %d families", len(pop), len(set(pop.families)))

    rand_state = SearchState(step=0, prompt=b"", loss=np.nan, best_prompt=b"", best_loss=np.nan, rng=run.rng)
    prompts = {
        "generic_passage": load_prompt(cfg, "generic_passage"),
        "handwritten_probe": load_prompt(cfg, "handwritten_probe"),
        "random_bytes": BestOfN("random_bytes", exp["random_length_bytes"], 1).propose(rand_state)[0],
        "random_tokens": BestOfN("random_tokens", exp["random_length_bytes"], 1,
                                 tokenizer=exp["random_tokenizer"]).propose(rand_state)[0],
    }
    for name, p in prompts.items():
        (run.out_dir / f"prompt_{name}.txt").write_bytes(p)

    extractor = FeatureExtractor.from_cfg(cfg)
    names = list(prompts)
    tls = extractor.token_logprobs(pop.specs, [prompts[n] for n in names])  # [model][prompt]

    def feats(name: str, dd: int) -> np.ndarray:
        j = names.index(name)
        return np.stack([chunk_features(row[j], dd) for row in tls])

    logp = np.log(pop.params_b)[:, None]
    baselines = {
        "log_params": zscore(logp),
        "passage_d1_bpb": feats("generic_passage", 1),
        f"passage_d{d}": feats("generic_passage", d),
        f"random_bytes_d{d}": feats("random_bytes", d),
        f"random_tokens_d{d}": feats("random_tokens", d),
        f"handwritten_d{d}": feats("handwritten_probe", d),
        f"passage_d{d}+log_params": np.hstack([feats("generic_passage", d), zscore(logp)]),
    }

    grid = lambda0_grid(cfg.readout)
    rows, preds = [], {}
    for name, Phi in baselines.items():
        pred, lams = nested_lofo(Phi, pop.S, pop.families, grid)
        preds[name] = pred
        summ = summarize(pred, pop.S, pop.families, pop.benchmarks)
        row = {"baseline": name, "dim": Phi.shape[1], "lambda0_median": float(np.median(lams)), **summ}
        for key, fn in (("spearman", mean_spearman), ("rmse", mean_rmse), ("within_spearman", mean_within_spearman)):
            pt, lo, hi = family_bootstrap(fn, pred, pop.S, pop.families, n_boot=exp["n_boot"], rng=run.rng)
            row.update({f"{key}/mean_ci_lo": lo, f"{key}/mean_ci_hi": hi})
        rows.append(row)
        log.info("%-28s spearman=%.3f [%.3f, %.3f]  rmse=%.2f", name, row["spearman/mean"],
                 row["spearman/mean_ci_lo"], row["spearman/mean_ci_hi"], row["rmse/mean"])

    # Gate 1: paired family bootstrap of (passage_d - log_params) mean Spearman
    a, b = preds[f"passage_d{d}"], preds["log_params"]
    K = pop.S.shape[1]
    diff_fn = lambda P, T, F: mean_spearman(P[:, :K], T, F) - mean_spearman(P[:, K:], T, F)  # noqa: E731
    diff, lo, hi = family_bootstrap(diff_fn, np.hstack([a, b]), pop.S, pop.families, n_boot=exp["n_boot"], rng=run.rng)
    noise = 100.0 * noise_ceiling_rmse(pop.S, pop.n_items)  # score points, like rmse/*
    passage_rmse = next(r["rmse/mean"] for r in rows if r["baseline"] == f"passage_d{d}")
    gate = {
        "gate1/spearman_gain": diff, "gate1/gain_ci_lo": lo, "gate1/gain_ci_hi": hi,
        "gate1/noise_ceiling_rmse_mean": float(np.mean(noise)),
        "gate1/passage_rmse_over_noise": float(passage_rmse / np.mean(noise)),
        "gate1/pass": bool(diff >= exp["gate_margin"] and lo > 0 and passage_rmse >= 1.5 * np.mean(noise)),
    }
    log.info("Gate 1: %s", json.dumps(gate, indent=1))

    table = pd.DataFrame(rows)
    table.to_csv(run.out_dir / "t1_baselines.csv", index=False)
    (run.out_dir / "gate1.json").write_text(json.dumps(gate, indent=1))
    run.logger.log_table("t1/baselines", list(table.columns), table.itertuples(index=False))
    run.logger.summary(gate | {"n_models": len(pop), "n_families": len(set(pop.families))})
    run.logger.finish()
    print(table[["baseline", "dim", "spearman/mean", "spearman/mean_ci_lo", "spearman/mean_ci_hi",
                 "rmse/mean", "within_spearman/mean"]].to_string(index=False))
    print(json.dumps(gate, indent=1))


if __name__ == "__main__":
    main()
