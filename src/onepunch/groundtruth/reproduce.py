"""Gate 0: do our lm-eval scores reproduce the archived leaderboard scores?"""

from __future__ import annotations

import numpy as np
import pandas as pd

from onepunch.data.benchmarks import sampling_se

Z = 2.0


def compare(scores: pd.DataFrame, protocol_version: str = "oll_v2", z: float = Z) -> pd.DataFrame:
    """Per (model_id, benchmark) with both an "ours" and an "archive" score.

    Rule (two-sided, independent estimates): pass iff

        |score_ours - score_archive| <= z * sqrt(se_ours**2 + se_archive**2),   z = 2

    A missing (NaN) standard error is replaced by the binomial sampling error
    sqrt(p(1-p)/n_items) at that source's score. Columns: model_id, benchmark, ours,
    archive, diff, se_ours, se_archive, tol, ratio (|diff|/tol), passed.
    """
    d = scores[scores["protocol_version"] == protocol_version].copy()
    n = d["n_items"].astype(float).clip(lower=1)
    d["stderr"] = d["stderr"].where(d["stderr"].notna(), sampling_se(d["score"], n))
    d = d.sort_values("date").drop_duplicates(["model_id", "benchmark", "source"], keep="last")
    ours = d[d["source"] == "ours"].set_index(["model_id", "benchmark"])
    arch = d[d["source"] == "archive"].set_index(["model_id", "benchmark"])
    j = ours[["score", "stderr"]].join(arch[["score", "stderr"]], how="inner", lsuffix="_o", rsuffix="_a")
    out = pd.DataFrame({
        "ours": j["score_o"], "archive": j["score_a"],
        "diff": j["score_o"] - j["score_a"],
        "se_ours": j["stderr_o"], "se_archive": j["stderr_a"],
    })
    out["tol"] = z * np.sqrt(out["se_ours"] ** 2 + out["se_archive"] ** 2)
    out["ratio"] = out["diff"].abs() / out["tol"].replace(0, np.nan)
    out["passed"] = out["diff"].abs() <= out["tol"]
    return out.reset_index()


def summarize(table: pd.DataFrame) -> dict:
    """Counts: pairs, failures, and failure list."""
    bad = table[~table["passed"]]
    return {"n_pairs": int(len(table)), "n_models": int(table["model_id"].nunique()),
            "n_failed": int(len(bad)), "failed": list(zip(bad["model_id"], bad["benchmark"]))}
