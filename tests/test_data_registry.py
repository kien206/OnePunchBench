import numpy as np
import pandas as pd
import pytest

from onepunch.data.registry import Registry, select_pool
from onepunch.types import REGISTRY_COLUMNS, ModelSpec


@pytest.fixture
def reg(population):
    return Registry(population.registry)


def test_basic_accessors(reg, population):
    assert len(reg) == len(population.specs)
    assert reg.families().tolist() == population.families.tolist()
    spec = reg.spec("org1/model1-2")
    assert isinstance(spec, ModelSpec)
    assert spec.family == "fam1" and spec.params_b == pytest.approx(1.5)
    assert [s.model_id for s in reg.specs()] == reg.model_ids()
    with pytest.raises(KeyError):
        reg.spec("nope/nope")


def test_validates_columns(population):
    with pytest.raises(ValueError):
        Registry(population.registry.drop(columns=["family"]))
    with pytest.raises(ValueError):
        Registry(population.registry.assign(extra=1))
    with pytest.raises(ValueError):
        Registry(pd.concat([population.registry, population.registry]))


def test_missing_optional_values_roundtrip(reg, tmp_path):
    df = reg.df.copy()
    df.loc[0, ["tokenizer_hash", "vocab_size", "params_b", "on_hub", "surrogate_group"]] = None
    r = Registry(df)
    assert pd.isna(r.df.loc[0, "params_b"]) and r.df.loc[0, "on_hub"] is pd.NA
    r.save(tmp_path / "sub" / "r.parquet")
    r2 = Registry.load(tmp_path / "sub" / "r.parquet")
    assert list(r2.df.columns) == list(REGISTRY_COLUMNS)
    assert r2.df.loc[0, "parents"] == [] and r2.df.loc[6, "parents"] == ["org1/model1-0"]
    assert r2.spec(r.df.loc[0, "model_id"]).params_b is None


def test_filters(reg):
    assert len(reg.filter(split="train")) == len(reg)
    assert len(reg.filter(split=["testA", "testB"])) == 0
    assert len(reg.filter(tier="S")) == len(reg)
    sub = reg.filter(min_params_b=1.0, max_params_b=2.0)
    assert set(sub.df["params_b"]) == {1.0, 1.5, 2.0}
    assert set(reg.filter(families=["fam0", "fam3"]).families()) == {"fam0", "fam3"}
    assert "fam0" not in set(reg.filter(exclude_families=["fam0"]).families())
    assert len(reg.filter(tokenizer_hash="tokA")) == 3 * 5  # fams 0, 3, 6
    assert len(reg.filter(tokenizer_hash=["tokA", "tokB"])) == 6 * 5
    assert len(reg.filter(archs=["LlamaForCausalLM"])) == len(reg)
    assert len(reg.filter(archs=["Nope"])) == 0
    assert len(reg.filter(on_hub=True)) == len(reg) and len(reg.filter(on_hub=False)) == 0
    assert len(reg.filter(moe=False)) == len(reg)
    ids = ["org0/model0-0", "org2/model2-1"]
    assert reg.filter(model_ids=ids).model_ids() == ids
    assert reg.filter(surrogate_group="tokA").families().tolist() == ["fam0"] * 5
    combo = reg.filter(split="train", families=["fam1", "fam2"], max_params_b=1.0)
    assert combo.model_ids() == ["org1/model1-0", "org1/model1-1", "org2/model2-0", "org2/model2-1"]
    with pytest.raises(TypeError):
        reg.filter(bogus=1)


def test_unknown_values_never_match(reg):
    df = reg.df.copy()
    df.loc[0, "on_hub"] = None
    df.loc[1, "params_b"] = np.nan
    r = Registry(df)
    assert len(r.filter(on_hub=True)) == len(r) - 1
    assert len(r.filter(max_params_b=100)) == len(r) - 1


def test_max_per_family_deterministic(reg):
    a = reg.filter(max_per_family=2, seed=0)
    assert len(a) == 8 * 2
    assert all(c == 2 for c in pd.Series(a.families()).value_counts())
    # same result for same seed, independent of row order
    shuffled = Registry(reg.df.sample(frac=1.0, random_state=3).reset_index(drop=True))
    assert set(shuffled.filter(max_per_family=2, seed=0).model_ids()) == set(a.model_ids())
    # different seed picks (at least somewhere) different models
    assert set(reg.filter(max_per_family=2, seed=1).model_ids()) != set(a.model_ids())
    # combines with other filters: cap applies after them
    b = reg.filter(max_params_b=1.0, max_per_family=1)
    assert len(b) == 8


def test_upsert(reg):
    base = reg.df.iloc[:2].copy()
    r = Registry(base)
    upd = base.iloc[:1].copy()
    upd["tokenizer_hash"] = "newhash"
    upd["family"] = None  # null never overwrites
    upd2 = base.iloc[:1].copy()
    upd2["model_id"] = "brand/new"
    r.upsert(pd.concat([upd, upd2]))
    assert len(r) == 3
    assert r.df.loc[0, "tokenizer_hash"] == "newhash" and r.df.loc[0, "family"] == "fam0"
    # prefer="old" only fills nulls
    old = base.iloc[:1].copy()
    old["tokenizer_hash"] = "other"
    old["vocab_size"] = 5
    r.df.loc[0, "vocab_size"] = pd.NA
    r.upsert(old, prefer="old")
    assert r.df.loc[0, "tokenizer_hash"] == "newhash" and r.df.loc[0, "vocab_size"] == 5
    # by_model: pinning main -> sha replaces the row
    pinned = base.iloc[:1].copy()
    pinned["revision"] = "abc123"
    r.upsert(pinned, by_model=True)
    assert len(r) == 3 and r.df.loc[0, "revision"] == "abc123"


def test_select_pool(reg):
    cfg = {"where": {"split": "train", "max_per_family": 1, "seed": 0}}
    assert len(select_pool(reg, cfg)) == 8
    ids = ["org0/model0-0", "org1/model1-0"]
    assert select_pool(reg, {"model_ids": ids, "where": {}}).model_ids() == ids
    both = select_pool(reg, {"model_ids": ids, "where": {"families": ["fam1"]}})
    assert both.model_ids() == ["org1/model1-0"]
    assert len(select_pool(reg, {})) == len(reg)
    from omegaconf import OmegaConf

    assert len(select_pool(reg, OmegaConf.create({"where": {"families": ["fam2"]}}))) == 5
    with pytest.raises(ValueError, match="unknown pool key"):
        select_pool(reg, {"filters": {}})


def test_select_pool_model_list(reg, tmp_path):
    (tmp_path / "m.yaml").write_text("models:\n  - model_id: org0/model0-0\n  - model_id: org1/model1-0\n")
    pool = select_pool(reg, {"model_list": "m.yaml", "where": {"families": ["fam1"]}}, root=tmp_path)
    assert pool.model_ids() == ["org1/model1-0"]
