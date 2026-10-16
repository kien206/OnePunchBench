import json

import numpy as np
import pandas as pd
import pytest

from onepunch.data.benchmarks import load_benchmarks, noise_ceiling_rmse, sampling_se
from onepunch.groundtruth.lmeval_runner import (
    LmEvalConfig, build_command, find_results, lmeval_bin, model_dir, parse_results)
from onepunch.groundtruth.reproduce import compare, summarize
from onepunch.types import SCORE_COLUMNS, ModelSpec

BENCH = load_benchmarks("data/benchmarks.yaml")

# shaped like lm-eval 0.4.x output for `--tasks leaderboard` (trimmed to the entries the parser reads)
RESULTS = {
    "results": {
        "leaderboard": {"alias": "leaderboard", "acc,none": 0.3, "acc_stderr,none": 0.003,
                        "prompt_level_strict_acc,none": 0.2, "prompt_level_strict_acc_stderr,none": 0.017},
        "leaderboard_bbh": {"alias": " - leaderboard_bbh", "acc_norm,none": 0.4, "acc_norm_stderr,none": 0.0066},
        "leaderboard_bbh_boolean_expressions": {"alias": "  - leaderboard_bbh_boolean_expressions",
                                                "acc_norm,none": 0.5, "acc_norm_stderr,none": 0.03},
        "leaderboard_gpqa": {"alias": " - leaderboard_gpqa", "acc_norm,none": 0.27, "acc_norm_stderr,none": "N/A"},
        "leaderboard_ifeval": {"alias": " - leaderboard_ifeval",
                               "prompt_level_strict_acc,none": 0.30, "prompt_level_strict_acc_stderr,none": 0.0197,
                               "inst_level_strict_acc,none": 0.42, "inst_level_strict_acc_stderr,none": "N/A",
                               "prompt_level_loose_acc,none": 0.33, "inst_level_loose_acc,none": 0.45},
        "leaderboard_math_hard": {"alias": " - leaderboard_math_hard", "exact_match,none": 0.05,
                                  "exact_match_stderr,none": 0.006},
        "leaderboard_mmlu_pro": {"alias": " - leaderboard_mmlu_pro", "acc,none": 0.15, "acc_stderr,none": 0.0033},
        # leaderboard_musr missing on purpose
    },
    "group_subtasks": {
        "leaderboard_gpqa": ["leaderboard_gpqa_diamond", "leaderboard_gpqa_extended", "leaderboard_gpqa_main"],
        "leaderboard_bbh": ["leaderboard_bbh_boolean_expressions"],
    },
    "n-samples": {
        "leaderboard_gpqa_diamond": {"original": 198, "effective": 198},
        "leaderboard_gpqa_extended": {"original": 546, "effective": 546},
        "leaderboard_gpqa_main": {"original": 448, "effective": 448},
        "leaderboard_ifeval": {"original": 541, "effective": 541},
        "leaderboard_bbh_boolean_expressions": {"original": 250, "effective": 250},
    },
    "config": {"model": "hf", "model_args": "pretrained=x/y,dtype=bfloat16"},
}


def test_parse_results_from_dict_and_file(tmp_path):
    df = parse_results(RESULTS, "x/y", "abc", BENCH, run_date="2026-01-01")
    assert list(df.columns) == list(SCORE_COLUMNS)
    d = df.set_index("benchmark")
    assert set(d.index) == {"ifeval", "bbh", "math_lvl5", "gpqa", "mmlu_pro"}  # musr absent -> skipped
    assert d.loc["bbh", "score"] == 0.4 and d.loc["bbh", "stderr"] == 0.0066
    assert d.loc["bbh", "n_items"] == 250  # taken from n-samples (here a trimmed fixture)
    assert d.loc["math_lvl5", "score"] == 0.05 and d.loc["mmlu_pro", "score"] == 0.15
    # IFEval = mean(prompt-level strict, inst-level strict); stderr falls back to sampling error
    assert d.loc["ifeval", "score"] == pytest.approx(0.36)
    assert d.loc["ifeval", "stderr"] == pytest.approx(sampling_se(0.36, 541))
    # "N/A" stderr -> binomial sampling error over the evaluated items
    assert d.loc["gpqa", "stderr"] == pytest.approx(sampling_se(0.27, 1192)) and d.loc["gpqa", "n_items"] == 1192
    assert (df["source"] == "ours").all() and (df["protocol_version"] == "oll_v2").all()
    assert (df["model_id"] == "x/y").all() and (df["date"] == "2026-01-01").all()
    path = tmp_path / "results_2026-01-01T00-00-00.json"
    path.write_text(json.dumps(RESULTS))
    pd.testing.assert_frame_equal(parse_results(path, "x/y", "abc", BENCH, run_date="2026-01-01"), df)


def test_find_results_newest(tmp_path):
    assert find_results(tmp_path / "none") is None
    d = tmp_path / "org__m" / "org__m"
    d.mkdir(parents=True)
    a, b = d / "results_2026-01-01.json", d / "results_2026-02-01.json"
    a.write_text("{}")
    b.write_text("{}")
    import os
    os.utime(a, (1, 1))
    assert find_results(tmp_path / "org__m") == b


def test_build_command(monkeypatch):
    monkeypatch.setenv("ONEPUNCH_LMEVAL_BIN", "/x/lm_eval")
    assert lmeval_bin() == "/x/lm_eval"
    spec = ModelSpec("Qwen/Qwen2.5-0.5B-Instruct", revision="abc123", dtype="bfloat16")
    cmd = build_command(spec, chat_template=True, cfg=LmEvalConfig(output_root="out"))
    assert cmd[0] == "/x/lm_eval"
    assert cmd[cmd.index("--model") + 1] == "hf"
    assert cmd[cmd.index("--model_args") + 1] == "pretrained=Qwen/Qwen2.5-0.5B-Instruct,revision=abc123,dtype=bfloat16"
    assert cmd[cmd.index("--tasks") + 1] == "leaderboard" and cmd[cmd.index("--batch_size") + 1] == "auto"
    assert cmd[cmd.index("--output_path") + 1] == str(model_dir(spec.model_id, "out")) == "out/Qwen__Qwen2.5-0.5B-Instruct"
    assert "--apply_chat_template" in cmd and "--fewshot_as_multiturn" in cmd and "--log_samples" not in cmd
    base = build_command(spec, chat_template=False, cfg=LmEvalConfig(log_samples=True))
    assert "--apply_chat_template" not in base and "--fewshot_as_multiturn" not in base and "--log_samples" in base
    trust = build_command(ModelSpec("a/b", trust_remote_code=True), False)
    assert "trust_remote_code=True" in trust[trust.index("--model_args") + 1] and "--trust_remote_code" in trust
    monkeypatch.delenv("ONEPUNCH_LMEVAL_BIN")
    assert lmeval_bin().endswith("onepunch-eval/bin/lm_eval")


def _scores(ours, archive, se_o=0.01, se_a=0.01):
    rows = []
    for src, val, se in (("ours", ours, se_o), ("archive", archive, se_a)):
        rows.append(dict(model_id="m", revision="main", benchmark="bbh", score=val, stderr=se, n_items=1000,
                         source=src, protocol_version="oll_v2", date="2026-01-01"))
    return pd.DataFrame(rows, columns=list(SCORE_COLUMNS))


def test_reproduction_rule():
    tol = 2 * np.sqrt(0.01**2 + 0.01**2)
    ok = compare(_scores(0.5 + 0.99 * tol, 0.5))
    assert ok["passed"].all() and ok["tol"].iloc[0] == pytest.approx(tol)
    bad = compare(_scores(0.5 - 1.01 * tol, 0.5))
    assert not bad["passed"].any() and summarize(bad)["failed"] == [("m", "bbh")]
    assert summarize(ok)["n_failed"] == 0
    # NaN stderr -> binomial fallback with n_items
    nan = compare(_scores(0.5, 0.5, se_o=np.nan, se_a=np.nan))
    assert nan["se_ours"].iloc[0] == pytest.approx(np.sqrt(0.25 / 1000))
    # no pair -> empty
    assert compare(_scores(0.5, 0.5).query("source == 'ours'")).empty


def test_benchmarks_file_and_noise():
    assert list(BENCH) == ["ifeval", "bbh", "math_lvl5", "gpqa", "musr", "mmlu_pro"]
    assert BENCH["ifeval"].metrics == ("prompt_level_strict_acc", "inst_level_strict_acc")
    assert BENCH["math_lvl5"].lm_eval_task == "leaderboard_math_hard" and BENCH["bbh"].n_shots == 3
    assert sampling_se(0.3, 448) == pytest.approx(np.sqrt(0.3 * 0.7 / 448))
    S = np.full((5, 2), 0.3)
    nc = noise_ceiling_rmse(S, [448, 12032])
    assert nc.shape == (2,) and nc[0] > nc[1] and nc[0] == pytest.approx(sampling_se(0.3, 448))
