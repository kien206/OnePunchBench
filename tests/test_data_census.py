import numpy as np
import pandas as pd
import pytest

from onepunch.data.benchmarks import load_benchmarks
from onepunch.data.census import build_registry_and_scores, clean_base_model, norm_id, resolve_roots
from onepunch.data.registry import Registry
from onepunch.data.scores import ScoreTable

BENCH = load_benchmarks("data/benchmarks.yaml")


def arch_row(name, base=None, merged=False, typ="🔶 fine-tuned on domain-specific datasets", params=0.5, scale=1.0,
             sub="2024-10-01", **kw):
    r = {
        "fullname": name, "Base Model": base if base is not None else name, "Merged": merged, "Type": typ,
        "MoE": False, "Architecture": "Qwen2ForCausalLM", "#Params (B)": params, "Precision": "bfloat16",
        "Chat Template": True, "Available on the hub": True, "Weight type": "Original",
        "Upload To Hub Date": "2024-09-01", "Submission Date": sub,
    }
    for i, b in enumerate(BENCH.values()):
        r[b.archive_raw_column] = (0.1 + 0.1 * i) * scale
    r.update(kw)
    return r


def test_clean_base_model():
    assert clean_base_model("Qwen/Qwen2.5-0.5B", "x/y") == "Qwen/Qwen2.5-0.5B"
    for raw in ["Removed", "", None, np.nan, "?", "x/y", "X/Y", "x/y (Merge)", "a/b c/d"]:
        assert clean_base_model(raw, "x/y") is None, raw


def test_resolve_roots_chain_case_and_roots():
    parents = {"c": ["B"], "b": ["a"], "A": [], "solo": []}
    r = resolve_roots(parents)
    assert r == {"c": "A", "b": "A", "A": "A", "solo": "solo"}  # case-insensitive, key spelling kept


def test_resolve_roots_unlisted_parent_is_root():
    assert resolve_roots({"m": ["Org/Base"]}) == {"m": "Org/Base"}


def test_resolve_roots_cycles_are_cut_deterministically():
    r = resolve_roots({"a": ["b"], "b": ["a"], "x": ["a"], "self": ["self"]})
    assert r["a"] == r["b"] == r["x"] == "a"
    assert r["self"] == "self"
    r2 = resolve_roots({"b": ["c"], "c": ["a"], "a": ["b"]})
    assert set(r2.values()) == {"a"}


def test_resolve_roots_merges():
    parents = {"m_ok": ["p1", "p2"], "p1": ["root"], "p2": ["root"], "m_bad": ["p1", "other"],
               "m_none": [], "child_of_none": ["m_none"], "m_single": ["p1"]}
    r = resolve_roots(parents, merged={"m_ok", "m_bad", "m_none", "m_single"})
    assert r["m_ok"] == "root" and r["m_single"] == "root"
    assert r["m_bad"] is None and r["m_none"] is None and r["child_of_none"] is None
    # same parent set but not flagged merged still needs a common root
    assert resolve_roots({"m": ["p1", "other"], "p1": []})["m"] is None


def small_archive():
    return pd.DataFrame([
        arch_row("Qwen/Qwen2.5-0.5B", typ="🟢 pretrained"),
        arch_row("Qwen/Qwen2.5-0.5B-Instruct", base="Qwen/Qwen2.5-0.5B", sub="2024-10-01"),
        arch_row("Qwen/Qwen2.5-0.5B-Instruct", base="Qwen/Qwen2.5-0.5B", sub="2024-11-01", scale=1.1),  # duplicate
        arch_row("user/ft", base="qwen/qwen2.5-0.5b-instruct"),          # case-insensitive chain
        arch_row("user/unknown_base", base="Removed"),
        arch_row("user/marker", base="user/marker (Merge)"),              # base unknown, not a merge
        arch_row("user/merge", base="user/merge (Merge)", merged=True),   # merge, parents unknown
        arch_row("user/merge_child", base="user/merge"),
        arch_row("user/cycle_a", base="user/cycle_b"),
        arch_row("user/cycle_b", base="user/cycle_a"),
        arch_row("user/adapter", base="Qwen/Qwen2.5-0.5B", **{"Weight type": "Adapter"}),
        arch_row("user/big", base="Qwen/Qwen2.5-0.5B", params=7.6),
        arch_row("user/noparams", base="Qwen/Qwen2.5-0.5B", params=-1.0, Architecture="?", Precision="float16"),
    ])


def test_build_registry_and_scores():
    reg_df, sc = build_registry_and_scores(small_archive(), BENCH)
    reg = Registry(reg_df)  # passes the column validation
    r = reg.df.set_index("model_id")
    assert len(reg) == 12  # duplicate fullname collapsed
    assert r.loc["Qwen/Qwen2.5-0.5B", "family"] == "qwen/qwen2.5-0.5b"
    assert r.loc["Qwen/Qwen2.5-0.5B-Instruct", "root_model"] == "Qwen/Qwen2.5-0.5B"
    assert r.loc["Qwen/Qwen2.5-0.5B-Instruct", "parents"] == ["Qwen/Qwen2.5-0.5B"]
    assert r.loc["user/ft", "family"] == "qwen/qwen2.5-0.5b" and r.loc["user/ft", "root_model"] == "Qwen/Qwen2.5-0.5B"
    assert r.loc["user/unknown_base", "family"] == "user/unknown_base" and r.loc["user/unknown_base", "split"] == "train"
    assert r.loc["user/marker", "merged"] == False and r.loc["user/marker", "parents"] == []  # noqa: E712
    # merges with unknown parents (and their descendants) are excluded; own family
    assert r.loc["user/merge", "merged"] == True and r.loc["user/merge", "split"] == "excluded"  # noqa: E712
    assert r.loc["user/merge", "root_model"] is None and r.loc["user/merge", "family"] == "user/merge"
    assert r.loc["user/merge_child", "split"] == "excluded"
    assert r.loc["user/cycle_a", "family"] == r.loc["user/cycle_b", "family"] == "user/cycle_a"
    assert r.loc["user/adapter", "loader"] == "unsupported_adapter" and r.loc["user/adapter", "split"] == "excluded"
    assert r.loc["user/big", "tier"] == "R" and r.loc["Qwen/Qwen2.5-0.5B", "tier"] == "S"
    assert np.isnan(r.loc["user/noparams", "params_b"]) and r.loc["user/noparams", "tier"] == "R"
    assert r.loc["user/noparams", "arch"] is None and r.loc["user/noparams", "dtype"] == "float16"
    assert r.loc["Qwen/Qwen2.5-0.5B", "chat_template"] == True and r.loc["Qwen/Qwen2.5-0.5B", "revision"] == "main"  # noqa: E712
    assert r.loc["Qwen/Qwen2.5-0.5B", "release_date"] == "2024-09-01"
    assert r["tokenizer_hash"].isna().all()


def test_hub_parents_rescue_merge():
    reg_df, _ = build_registry_and_scores(small_archive(), BENCH, parents_extra={"user/merge": ["Qwen/Qwen2.5-0.5B", "user/ft"]})
    r = reg_df.set_index("model_id")
    assert r.loc["user/merge", "root_model"] == "Qwen/Qwen2.5-0.5B" and r.loc["user/merge", "split"] == "train"
    assert r.loc["user/merge_child", "split"] == "train"


def test_scores_raw_scale_and_stderr():
    reg_df, sc = build_registry_and_scores(small_archive(), BENCH)
    table = ScoreTable(sc)  # validates [0, 1]
    assert len(sc) == 12 * len(BENCH)
    inst = table.df[(table.df.model_id == "Qwen/Qwen2.5-0.5B-Instruct") & (table.df.benchmark == "bbh")].iloc[0]
    assert inst["score"] == pytest.approx(0.2 * 1.1)  # latest duplicate (submission 2024-11-01)
    assert inst["source"] == "archive" and inst["protocol_version"] == "oll_v2" and inst["n_items"] == BENCH["bbh"].n_items
    assert inst["stderr"] == pytest.approx(np.sqrt(inst["score"] * (1 - inst["score"]) / BENCH["bbh"].n_items))
    assert set(sc["benchmark"]) == set(BENCH)


def test_percent_columns_are_converted():
    a = small_archive()
    for b in BENCH.values():
        a[b.archive_raw_column] = a[b.archive_raw_column] * 100
    _, sc = build_registry_and_scores(a, BENCH)
    assert sc["score"].max() <= 1.0
    row = sc[(sc.model_id == "Qwen/Qwen2.5-0.5B") & (sc.benchmark == "ifeval")].iloc[0]
    assert row["score"] == pytest.approx(0.1)


def test_missing_scores_dropped():
    a = small_archive()
    a.loc[0, BENCH["gpqa"].archive_raw_column] = np.nan
    _, sc = build_registry_and_scores(a, BENCH)
    assert not ((sc.model_id == "Qwen/Qwen2.5-0.5B") & (sc.benchmark == "gpqa")).any()
    assert norm_id(" Qwen/X ") == "qwen/x"


@pytest.mark.slow
def test_real_archive_snapshot():
    """Uses the snapshot written by scripts/pull_open_llm_leaderboard.py (skipped if absent)."""
    from pathlib import Path

    snap = Path("data/archive/oll_v2_contents.parquet")
    if not snap.exists():
        pytest.skip("no archive snapshot")
    a = pd.read_parquet(snap)
    reg_df, sc = build_registry_and_scores(a, BENCH)
    Registry(reg_df)
    ScoreTable(sc)
    assert len(reg_df) == a["fullname"].nunique()
    assert sc["score"].between(0, 1).all()
