"""Print the test metrics of a train.py run as a table (RMSE in score points, Spearman).

    python scripts/report_metrics.py runs/<date>/<time>_train          # one run
    python scripts/report_metrics.py runs/2026-10-17/*_train           # several runs, one block each
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def report(run_dir: Path) -> None:
    m = json.loads((run_dir / "metrics.json").read_text())
    splits = sorted({k.split("/")[1] for k in m if k.startswith("eval/")})
    benches = sorted({k.split("/")[3] for k in m if k.startswith("eval/") and k.count("/") == 3
                      and k.split("/")[2] == "rmse" and k.split("/")[3] != "mean"})
    print(f"== {run_dir}  (steps run: {m.get('steps_run')}, prompt changed: {m.get('prompt_changed')})")
    head = f"{'split':8s} {'n':>4s} {'skip':>4s} {'RMSE':>6s} {'Spearman':>8s} {'[95% CI]':>16s}  " + \
           " ".join(f"{b[:8]:>8s}" for b in benches)
    print(head + "\n" + "-" * len(head))
    for s in splits:
        g = lambda k, d=float("nan"): m.get(f"eval/{s}/{k}", d)  # noqa: E731
        if f"eval/{s}/error" in m:
            print(f"{s:8s} failed: {m[f'eval/{s}/error']}")
            continue
        ci = f"[{g('spearman/mean_ci_lo'):.2f}, {g('spearman/mean_ci_hi'):.2f}]"
        print(f"{s:8s} {int(g('n_models', 0)):4d} {int(g('n_failed', 0)):4d} {g('rmse/mean'):6.2f} "
              f"{g('spearman/mean'):8.3f} {ci:>16s}  " + " ".join(f"{g('rmse/' + b):8.2f}" for b in benches))
    print("(RMSE in score points on the 0-100 scale; per-benchmark columns are RMSE)\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+", type=Path)
    for d in ap.parse_args().run_dirs:
        report(d)


if __name__ == "__main__":
    main()
