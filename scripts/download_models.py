"""Prefetch or evict the weights of a pool (safetensors + config + tokenizer files only).

    python scripts/download_models.py --pool smoke                    # download
    python scripts/download_models.py --pool tier_s --max-gb 150      # stop before exceeding the budget
    python scripts/download_models.py --pool smoke --evict            # remove from the HF cache
    python scripts/download_models.py --pool tier_s --dry-run         # sizes only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


import onepunch  # noqa: F401,E402  (loads .env: HF_TOKEN, HF_HOME) before huggingface_hub
from huggingface_hub import HfApi, scan_cache_dir, snapshot_download  # noqa: E402

from onepunch.data.registry import Registry, select_pool  # noqa: E402

PATTERNS = ["*.safetensors", "*.safetensors.index.json", "*.json", "tokenizer*", "*.model", "*.txt", "*.tiktoken"]


def pool_cfg(pool: str) -> dict:
    p = Path(pool)
    if not p.exists():
        p = Path("configs/pool") / f"{pool}.yaml"
    return yaml.safe_load(p.read_text()) or {}


def planned_bytes(api: HfApi, spec) -> int | None:
    """Bytes of the files snapshot_download would fetch; None if the repo has no safetensors."""
    from fnmatch import fnmatch

    info = api.model_info(spec.model_id, revision=spec.revision, files_metadata=True)
    files = [(s.rfilename, s.size or 0) for s in info.siblings]
    if not any(f.endswith(".safetensors") for f, _ in files):
        return None
    return sum(sz for f, sz in files if any(fnmatch(f, p) for p in PATTERNS))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default="data/registry/registry.parquet")
    ap.add_argument("--pool", required=True, help="pool name (configs/pool/<name>.yaml) or yaml path")
    ap.add_argument("--max-gb", type=float, default=None, help="skip models that would exceed this total")
    ap.add_argument("--evict", action="store_true", help="delete the pool's cached revisions instead")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    specs = select_pool(Registry.load(args.registry), pool_cfg(args.pool)).specs()
    print(f"pool {args.pool}: {len(specs)} models")

    if args.evict:
        ids = {s.model_id for s in specs}
        info = scan_cache_dir()
        hashes = [rev.commit_hash for repo in info.repos if repo.repo_type == "model" and repo.repo_id in ids
                  for rev in repo.revisions]
        strat = info.delete_revisions(*hashes)
        print(f"evicting {len(hashes)} revisions, freeing {strat.expected_freed_size_str}")
        if not args.dry_run:
            strat.execute()
        return

    api = HfApi()
    total = 0.0
    n_done = 0
    for s in specs:
        try:
            size = planned_bytes(api, s)
        except Exception as e:
            print(f"  skip {s.model_id}: {type(e).__name__}")
            continue
        if size is None:
            print(f"  skip {s.model_id}: no safetensors")
            continue
        if args.max_gb is not None and (total + size) / 1e9 > args.max_gb:
            print(f"  skip {s.model_id}: {size / 1e9:.1f} GB would exceed --max-gb {args.max_gb}")
            continue
        total += size
        if not args.dry_run:
            try:
                snapshot_download(s.model_id, revision=s.revision, allow_patterns=PATTERNS)
            except Exception as e:  # gated / removed
                total -= size
                print(f"  failed {s.model_id}: {type(e).__name__}")
                continue
        n_done += 1
        print(f"  {'would fetch' if args.dry_run else 'fetched'} {s.model_id} ({size / 1e9:.2f} GB)")
    print(f"{n_done}/{len(specs)} models, total {total / 1e9:.1f} GB")


if __name__ == "__main__":
    main()
