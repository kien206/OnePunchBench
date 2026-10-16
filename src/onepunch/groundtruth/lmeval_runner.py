"""Run lm-eval's `leaderboard` suite (OLL v2 protocol) and parse its results.

lm-eval lives in a separate conda env (its `math` extra conflicts with Hydra's antlr pin),
so it is only ever called through a subprocess; its path is `$ONEPUNCH_LMEVAL_BIN`.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from onepunch.data.benchmarks import Benchmark, sampling_se
from onepunch.types import SCORE_COLUMNS, ModelSpec

DEFAULT_LMEVAL_BIN = "/media/volume/DoubleSteeringLLM/miniconda3/envs/onepunch-eval/bin/lm_eval"
PROTOCOL = "oll_v2"


def lmeval_bin() -> str:
    return os.environ.get("ONEPUNCH_LMEVAL_BIN", DEFAULT_LMEVAL_BIN)


@dataclass
class LmEvalConfig:
    tasks: str = "leaderboard"
    batch_size: str = "auto"
    output_root: str = "data/lmeval_runs"
    log_samples: bool = False
    device: str | None = None
    extra_args: list[str] = field(default_factory=list)


def model_dir(model_id: str, output_root: str | Path) -> Path:
    """data/lmeval_runs/<org>__<name>; lm-eval then nests its own subfolder inside."""
    return Path(output_root) / model_id.replace("/", "__")


def build_command(spec: ModelSpec, chat_template: bool, cfg: LmEvalConfig | None = None) -> list[str]:
    """lm-eval command for one model under the pinned OLL v2 protocol.

    HF backend, the dtype and revision of the spec, task defaults for shots,
    chat template + multi-turn few-shot iff `chat_template` (the archive's column).
    """
    cfg = cfg or LmEvalConfig()
    model_args = f"pretrained={spec.model_id},revision={spec.revision},dtype={spec.dtype}"
    if spec.trust_remote_code:
        model_args += ",trust_remote_code=True"
    cmd = [lmeval_bin(), "--model", "hf", "--model_args", model_args, "--tasks", cfg.tasks,
           "--batch_size", cfg.batch_size, "--output_path", str(model_dir(spec.model_id, cfg.output_root))]
    if chat_template:
        cmd += ["--apply_chat_template", "--fewshot_as_multiturn"]
    if spec.trust_remote_code:
        cmd += ["--trust_remote_code"]
    if cfg.log_samples:
        cmd += ["--log_samples"]
    if cfg.device:
        cmd += ["--device", cfg.device]
    return cmd + list(cfg.extra_args)


def run_lmeval(cmd: list[str], log_path: str | Path | None = None) -> int:
    """Run lm-eval as a subprocess (stdout/stderr to log_path if given); return its exit code."""
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "ab") as f:
            return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
    return subprocess.run(cmd).returncode


def find_results(out_dir: str | Path) -> Path | None:
    """Newest results_*.json below out_dir (lm-eval writes <out>/<model>/results_<timestamp>.json)."""
    files = sorted(Path(out_dir).rglob("results_*.json"), key=lambda p: (p.stat().st_mtime, p.name))
    return files[-1] if files else None


def _num(x: Any) -> float:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else float("nan")


def _metric(res: Mapping[str, Any], name: str) -> tuple[float, float]:
    """(value, stderr) of one metric of an lm-eval result block; stderr NaN if "N/A"/absent."""
    return _num(res.get(f"{name},none")), _num(res.get(f"{name}_stderr,none"))


def parse_results(
    results: Mapping[str, Any] | str | Path,
    model_id: str,
    revision: str,
    benchmarks: Mapping[str, Benchmark],
    run_date: str | None = None,
) -> pd.DataFrame:
    """lm-eval 0.4 results JSON -> long-format score rows (source="ours").

    Per benchmark the block `results[bench.lm_eval_task]` (the group aggregate) is read.
    score = mean of bench.metrics (ifeval: prompt_level_strict_acc and inst_level_strict_acc);
    stderr = lm-eval's stderr if the benchmark has one metric and it is numeric, else
    the binomial sampling error with n_items. Benchmarks missing from the JSON are skipped.
    """
    if not isinstance(results, Mapping):
        results = json.loads(Path(results).read_text())
    blocks = {**(results.get("groups") or {}), **(results.get("results") or {})}
    n_samples = results.get("n-samples") or {}
    run_date = run_date or date.today().isoformat()
    rows = []
    for name, b in benchmarks.items():
        blk = blocks.get(b.lm_eval_task)
        if blk is None:
            continue
        vals = [_metric(blk, m) for m in b.metrics]
        if any(np.isnan(v) for v, _ in vals):
            continue
        score = float(np.mean([v for v, _ in vals]))
        se = vals[0][1] if len(vals) == 1 else float("nan")
        n = b.n_items
        leaves = (results.get("group_subtasks") or {}).get(b.lm_eval_task) or [b.lm_eval_task]
        sub = [n_samples[t] for t in leaves if t in n_samples]
        if sub:  # items actually evaluated (smaller than the full set under --limit)
            n = int(sum(s.get("effective", s.get("original", 0)) for s in sub))
        if np.isnan(se):
            se = float(sampling_se(score, n))
        rows.append({"model_id": model_id, "revision": revision, "benchmark": name, "score": score,
                     "stderr": se, "n_items": n, "source": "ours", "protocol_version": PROTOCOL,
                     "date": run_date})
    return pd.DataFrame(rows, columns=list(SCORE_COLUMNS))
