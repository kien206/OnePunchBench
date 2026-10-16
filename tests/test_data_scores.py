import numpy as np
import pandas as pd
import pytest

from onepunch.data.scores import ScoreTable
from onepunch.types import SCORE_COLUMNS


def row(model, bench, score, source="archive", date="2024-06-01", rev="main", proto="oll_v2", se=0.01, n=100):
    return dict(model_id=model, revision=rev, benchmark=bench, score=score, stderr=se, n_items=n,
                source=source, protocol_version=proto, date=date)


def table(*rows):
    return ScoreTable(pd.DataFrame(list(rows), columns=list(SCORE_COLUMNS)))


def test_population_roundtrip(population, tmp_path):
    t = ScoreTable(population.scores)
    assert len(t) == len(population.scores)
    t.save(tmp_path / "s" / "scores.parquet")
    t2 = ScoreTable.load(tmp_path / "s" / "scores.parquet")
    ids = [s.model_id for s in population.specs]
    S, mask = t2.matrix(ids, population.benchmarks)
    assert mask.all()
    np.testing.assert_allclose(S, population.S)


def test_validate_range_and_columns():
    with pytest.raises(ValueError):
        table(row("a", "bbh", 1.5))
    with pytest.raises(ValueError):
        table(row("a", "bbh", -0.1))
    with pytest.raises(ValueError):
        table(row("a", "bbh", 0.5, source="other"))
    with pytest.raises(ValueError):
        ScoreTable(pd.DataFrame({"model_id": ["a"]}))


def test_append_dedupe_newest_wins():
    t = table(row("a", "bbh", 0.3, date="2024-06-01"))
    t.append(pd.DataFrame([row("a", "bbh", 0.4, date="2025-01-01"), row("a", "gpqa", 0.2)]))
    assert len(t) == 2
    assert t.df[t.df.benchmark == "bbh"]["score"].item() == 0.4
    # an older row added later does not replace a newer one
    t.append(pd.DataFrame([row("a", "bbh", 0.9, date="2020-01-01")]))
    assert t.df[t.df.benchmark == "bbh"]["score"].item() == 0.4
    # same date: the later row wins
    t.append(pd.DataFrame([row("a", "bbh", 0.5, date="2025-01-01")]))
    assert t.df[t.df.benchmark == "bbh"]["score"].item() == 0.5
    # different source or revision is a different key
    t.append(pd.DataFrame([row("a", "bbh", 0.6, source="ours"), row("a", "bbh", 0.7, rev="sha1")]))
    assert len(t) == 4


def test_matrix_priority_nan_mask():
    t = table(
        row("a", "bbh", 0.30, source="archive"), row("a", "bbh", 0.35, source="ours"),
        row("a", "gpqa", 0.25, source="archive"),
        row("b", "bbh", 0.50, source="archive"),
        row("c", "bbh", 0.60, source="archive", proto="other_protocol"),
    )
    S, mask = t.matrix(["a", "b", "c", "zzz"], ["bbh", "gpqa"])
    assert S.dtype == np.float64 and S.shape == (4, 2)
    np.testing.assert_allclose(S[0], [0.35, 0.25])  # ours beats archive, archive fills gpqa
    assert S[1, 0] == 0.50 and np.isnan(S[1, 1])
    assert np.isnan(S[2]).all() and np.isnan(S[3]).all()  # wrong protocol / unknown model
    np.testing.assert_array_equal(mask, ~np.isnan(S))
    S2, _ = t.matrix(["a"], ["bbh"], sources=("archive", "ours"))
    assert S2[0, 0] == 0.30
    S3, m3 = t.matrix(["a"], ["bbh"], sources=("archive",))
    assert S3[0, 0] == 0.30 and m3.all()
    S4, _ = t.matrix(["c"], ["bbh"], protocol_version="other_protocol")
    assert S4[0, 0] == 0.60


def test_matrix_ignores_revision_and_takes_newest():
    t = table(row("a", "bbh", 0.2, rev="old", date="2024-01-01"), row("a", "bbh", 0.3, rev="new", date="2025-01-01"))
    S, _ = t.matrix(["a"], ["bbh"])
    assert S[0, 0] == 0.3
