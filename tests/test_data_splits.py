import numpy as np
import pandas as pd
import pytest

from onepunch.data.splits import assign_splits, canonical_family, load_splits_config

CFG = {
    "cutoff_date": "2025-03-13",
    "family_aliases": {"org7/*": "alias7"},
    "testC": {"archs": ["*Mamba*", "*Moe*"], "use_moe_flag": True},
    "excluded_archs": ["T5*"],
    "val": ["fam1"], "calib": ["fam2"], "testA": ["fam3"],
    "train_min_models": 3,
    "surrogates": {"grpA": ["org0/model0-0"]},
    "overrides": {},
}


@pytest.fixture
def df(population):
    d = population.registry.copy()
    d["split"] = "train"
    d["family"] = None
    d["root_model"] = None  # families come from the parent links
    return d


def split_of(out, fam):
    s = out.loc[out["family"] == fam, "split"].unique()
    assert len(s) == 1, f"family {fam} spans {s}"
    return s[0]


def test_basic_assignment_and_family_chain(df):
    out = assign_splits(df, CFG)
    assert list(out.columns) == list(df.columns)
    assert out["root_model"].iloc[0] == "org0/model0-0"
    assert set(out["family"]) == {"org0/model0-0", "org1/model1-0", "org2/model2-0", "org3/model3-0",
                                  "org4/model4-0", "org5/model5-0", "org6/model6-0", "alias7"}
    # patterns match the lowercased family name, so use explicit ids here
    cfg = {**CFG, "val": ["org1/model1-0"], "calib": ["org2/model2-0"], "testA": ["org3/model3-0"]}
    out = assign_splits(df, cfg)
    assert split_of(out, "org1/model1-0") == "val"
    assert split_of(out, "org2/model2-0") == "calib"
    assert split_of(out, "org3/model3-0") == "testA"
    assert split_of(out, "org0/model0-0") == "train"
    assert split_of(out, "alias7") == "train"
    assert out["split"].isin(["train", "val", "calib", "testA"]).all()


def test_deterministic_and_idempotent(df):
    a = assign_splits(df, CFG)
    b = assign_splits(df.sample(frac=1.0, random_state=1).reset_index(drop=True), CFG)
    pd.testing.assert_frame_equal(a.sort_values("model_id").reset_index(drop=True),
                                  b.sort_values("model_id").reset_index(drop=True))
    c = assign_splits(a, CFG)  # re-running on its own output changes nothing
    pd.testing.assert_frame_equal(a, c)


def test_families_never_span_splits(df):
    cfg = {**CFG, "val": ["org1/model1-0"], "testC": {"archs": ["*Mamba*"], "use_moe_flag": True}}
    d = df.copy()
    d.loc[d["family"].isna() & d["model_id"].str.startswith("org4"), "arch"] = "MambaForCausalLM"
    d.loc[d["model_id"] == "org5/model5-1", "moe"] = True  # single MoE model inside a dense family
    d.loc[d["model_id"] == "org6/model6-0", "release_date"] = "2025-06-01"  # family min date stays old
    out = assign_splits(d, cfg)
    for fam, g in out.groupby("family"):
        assert g.loc[g["split"] != "excluded", "split"].nunique() <= 1
    assert split_of(out, "org4/model4-0") == "testC"
    # lone MoE model in a dense train family is excluded, not split off
    row = out[out["model_id"] == "org5/model5-1"].iloc[0]
    assert row["split"] == "excluded"
    assert (out[out["family"] == row["family"]]["split"] == "train").sum() == 4


def test_release_date_test_b(df):
    d = df.copy()
    fam3 = d["model_id"].str.startswith("org3")
    d.loc[fam3, "release_date"] = "2025-06-01"
    d.loc[d["model_id"] == "org0/model0-1", "release_date"] = "2025-06-01"  # new model in an old family
    out = assign_splits(d, CFG)
    assert split_of(out, "org3/model3-0") == "testB"
    assert split_of(out, "org0/model0-0") == "train"  # family first released before cutoff


def test_small_family_and_excluded(df):
    d = df.copy()
    d.loc[d["model_id"].str.startswith("org0"), "on_hub"] = False
    d.loc[d["model_id"] == "org1/model1-0", "loader"] = "unsupported_adapter"
    d.loc[d["model_id"] == "org2/model2-0", "arch"] = "T5ForConditionalGeneration"
    out = assign_splits(d, CFG)
    assert split_of(out, "org0/model0-0") == "excluded"  # < train_min_models on the hub
    o = out.set_index("model_id")["split"]
    assert o["org1/model1-0"] == "excluded" and o["org2/model2-0"] == "excluded" and o["org1/model1-1"] == "train"


def _set(d, model_id, **kw):
    i = d.index[d["model_id"] == model_id][0]
    for k, v in kw.items():
        d.at[i, k] = v


def test_merge_exclusion(df):
    d = df.copy()
    d["merged"] = d["merged"].astype(object)
    # two parents with a common root -> kept; parents with different roots -> excluded;
    # merge without known parents -> excluded, and so is its child
    _set(d, "org0/model0-1", merged=True, parents=["org0/model0-2", "org0/model0-3"])
    _set(d, "org1/model1-1", merged=True, parents=["org1/model1-2", "org2/model2-2"])
    _set(d, "org1/model1-2", merged=True, parents=[])
    _set(d, "org1/model1-3", parents=["org1/model1-2"])
    out = assign_splits(d.assign(merged=d["merged"].astype(bool)), CFG).set_index("model_id")
    assert out.loc["org0/model0-1", "split"] == "train" and out.loc["org0/model0-1", "root_model"] == "org0/model0-0"
    assert out.loc["org1/model1-1", "split"] == "excluded" and out.loc["org1/model1-1", "family"] == "org1/model1-1"
    assert out.loc["org1/model1-2", "split"] == "excluded"
    assert out.loc["org1/model1-3", "split"] == "excluded"


def test_extra_parents_from_hub_cache(df):
    d = df[df["model_id"] == "org0/model0-1"].copy()
    d["parents"] = [["hub/derived"]]
    d["root_model"] = None
    out = assign_splits(d, CFG, extra_parents={"hub/derived": ["hub/root"], "hub/root": []})
    assert out["root_model"].iloc[0] == "hub/root" and out["family"].iloc[0] == "hub/root"


def test_surrogate_groups(df):
    out = assign_splits(df, CFG)
    tok = out.loc[out["model_id"] == "org0/model0-0", "tokenizer_hash"].iloc[0]
    got = out[out["surrogate_group"] == "grpA"]
    assert set(got["tokenizer_hash"]) == {tok} and (got["split"] == "train").all()
    assert len(got) == (out["tokenizer_hash"] == tok).sum()
    d = df.copy()
    d.loc[d["model_id"].str.startswith("org6"), "arch"] = "T5ForConditionalGeneration"  # excluded -> no group
    out2 = assign_splits(d, CFG)
    assert out2.loc[out2["split"] == "excluded", "surrogate_group"].isna().all()


def test_canonical_family_and_real_config():
    assert canonical_family("Meta-Llama/Llama-3.2-1B", {"meta-llama/llama-3*": "llama3"}) == "llama3"
    assert canonical_family("Qwen/Qwen2.5-0.5B") == "qwen/qwen2.5-0.5b"
    cfg = load_splits_config("data/splits.yaml")
    assert {"val", "calib", "testA", "cutoff_date"} <= set(cfg)
    listed = [p for k in ("val", "calib", "testA") for p in cfg[k]]
    assert len(listed) == len(set(listed))
