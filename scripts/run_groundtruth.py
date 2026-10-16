"""Score models with the lm-eval OLL v2 protocol and append source="ours" rows to scores.parquet.

    python scripts/run_groundtruth.py --pool smoke --dry-run            # print commands only
    python scripts/run_groundtruth.py --model-ids Qwen/Qwen2.5-0.5B     # run + parse
    python scripts/run_groundtruth.py --pool smoke --parse-only         # parse existing results

Existing results (data/lmeval_runs/<model>/**/results_*.json) are parsed instead of re-run.
lm-eval is called through $ONEPUNCH_LMEVAL_BIN (default: the onepunch-eval conda env).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


import onepunch  # noqa: F401,E402  (loads .env: HF_TOKEN, HF_HOME) before huggingface_hub
from onepunch.data.benchmarks import load_benchmarks  # noqa: E402
from onepunch.data.registry import Registry, select_pool  # noqa: E402
from onepunch.data.scores import ScoreTable  # noqa: E402
from onepunch.groundtruth.lmeval_runner import (  # noqa: E402
    LmEvalConfig, build_command, find_results, model_dir, parse_results, run_lmeval)


def load_pool_cfg(pool: str) -> dict:
    p = Path(pool)
    if not p.exists():
        p = Path("configs/pool") / f"{pool}.yaml"
    return yaml.safe_load(p.read_text()) or {}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default="data/registry/registry.parquet")
    ap.add_argument("--scores", default="data/scores/scores.parquet")
    ap.add_argument("--benchmarks", default="data/benchmarks.yaml")
    ap.add_argument("--out-dir", default="data/lmeval_runs")
    ap.add_argument("--pool", help="pool name (configs/pool/<name>.yaml) or yaml path")
    ap.add_argument("--model-ids", nargs="*")
    ap.add_argument("--max-models", type=int)
    ap.add_argument("--log-samples", action="store_true")
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    ap.add_argument("--parse-only", action="store_true", help="never launch lm-eval")
    args = ap.parse_args(argv)

    reg = Registry.load(args.registry)
    if args.pool:
        reg = select_pool(reg, load_pool_cfg(args.pool))
    if args.model_ids:
        reg = reg.filter(model_ids=args.model_ids)
    df = reg.df.head(args.max_models) if args.max_models else reg.df
    benchmarks = load_benchmarks(args.benchmarks)
    cfg = LmEvalConfig(batch_size=args.batch_size, output_root=args.out_dir, log_samples=args.log_samples)
    table = ScoreTable.load(args.scores) if Path(args.scores).exists() else ScoreTable()

    n_new = 0
    for _, r in df.iterrows():
        spec = Registry(df[df["model_id"] == r["model_id"]]).spec(r["model_id"])
        out = model_dir(spec.model_id, args.out_dir)
        res = find_results(out)
        if res is None:
            cmd = build_command(spec, chat_template=bool(r["chat_template"]), cfg=cfg)
            if args.dry_run:
                print(" ".join(cmd))
                continue
            if args.parse_only:
                print(f"no results for {spec.model_id}")
                continue
            code = run_lmeval(cmd, log_path=out / "lm_eval.log")
            if code != 0:
                print(f"lm-eval failed for {spec.model_id} (exit {code}); see {out / 'lm_eval.log'}")
                continue
            res = find_results(out)
            if res is None:
                print(f"no results json for {spec.model_id}")
                continue
        rows = parse_results(res, spec.model_id, spec.revision, benchmarks)
        table.append(rows)
        n_new += len(rows)
        print(f"{spec.model_id}: {len(rows)} benchmark scores from {res}")
    if not args.dry_run and n_new:
        table.save(args.scores)
        print(f"wrote {n_new} rows -> {args.scores}")


if __name__ == "__main__":
    main()
