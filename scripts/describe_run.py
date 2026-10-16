"""Show what a train.py run would use, without running it: benchmarks, score source, the
models of the pool per role (train / monitor / eval), and which weights are already downloaded.
Takes the same overrides as train.py.

    python scripts/describe_run.py                    # default config (pool=smoke)
    python scripts/describe_run.py pool=tier_s
    python scripts/describe_run.py pool=tier_r roles.eval=[testA] --models   # also list every model
"""

from __future__ import annotations

import sys
from collections import Counter

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

import onepunch  # noqa: F401  (loads .env)
from onepunch.data.registry import Registry, select_pool
from onepunch.utils.experiment import load_population

SHOW_MODELS = "--models" in sys.argv
if SHOW_MODELS:
    sys.argv.remove("--models")


def cached_repos() -> set[tuple[str, str]]:
    """(repo_id, commit or ref) pairs whose safetensors weights are in the HF cache."""
    from huggingface_hub import scan_cache_dir

    try:
        info = scan_cache_dir()
    except Exception:  # empty / missing cache
        return set()
    out = set()
    for repo in info.repos:
        for rev in repo.revisions:
            if any(f.file_name.endswith(".safetensors") for f in rev.files):
                out.add((repo.repo_id, rev.commit_hash))
                out.update((repo.repo_id, ref) for ref in rev.refs)
    return out


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    pool_name = HydraConfig.get().runtime.choices.get("pool")
    pool = select_pool(Registry.load(cfg.paths.registry), cfg.pool)
    pop = load_population(cfg)  # pool models with every benchmark score
    cached = cached_repos()
    is_cached = lambda s: (s.model_id, s.revision) in cached  # noqa: E731

    print(f"benchmarks : {list(cfg.scores.benchmarks)}")
    print(f"scores from: {list(cfg.scores.sources)} (protocol {cfg.scores.protocol_version})")
    print(f"pool       : {pool_name}: {cfg.pool.get('description', '')}")
    print(f"             {len(pool)} models selected, {len(pop)} with all benchmark scores")

    used: set[str] = set()
    for role in ("train", "monitor", "eval"):
        splits = list(cfg.roles.get(role) or [])
        used.update(splits)
        sub = pop.where(splits)
        print(f"\n[{role}] splits={splits}: {len(sub)} models, {len(set(sub.families))} families, "
              f"{len(set(sub.tokenizers))} tokenizers, weights cached {sum(map(is_cached, sub.specs))}/{len(sub)}")
        for split in splits:
            part = pop.where([split])
            fams = Counter(part.families.tolist())
            print(f"  {split:8s} {len(part):4d} models" +
                  (" | " + ", ".join(f"{f} ({n})" for f, n in fams.most_common()) if len(part) else ""))
            if SHOW_MODELS:
                for s in part.specs:
                    print(f"      {s.model_id:60s} {s.params_b or float('nan'):6.2f}B  "
                          f"{'cached' if is_cached(s) else 'download'}")

    unused = Counter(s for s in pop.splits.tolist() if s not in used)
    if unused:
        print(f"\nin the pool but in no role (never loaded): {dict(unused)}")
    missing = len(pop) - sum(map(is_cached, pop.specs))
    if missing:
        print(f"\n{missing} models are not downloaded yet. train.py downloads them on first use, or prefetch with\n"
              f"  python scripts/download_models.py --pool {pool_name}\nthen set HF_HUB_OFFLINE=1 in .env.")


if __name__ == "__main__":
    main()
