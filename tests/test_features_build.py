"""features._target_ selects the FeatureSource; the default stays the chunked log-prob extractor."""

from pathlib import Path

import numpy as np
from hydra import compose, initialize_config_dir

from onepunch.features import FeatureExtractor, ParamFeatures, build_feature_source

CONFIGS = str(Path(__file__).resolve().parents[1] / "configs")


def _cfg(*overrides):
    with initialize_config_dir(config_dir=CONFIGS, version_base="1.3"):
        return compose("config", overrides=list(overrides))


def test_default_is_logprob_extractor():
    src = build_feature_source(_cfg())
    assert isinstance(src, FeatureExtractor) and src.d == 16


def test_log_params_source(population):
    src = build_feature_source(_cfg("features=log_params"))
    assert isinstance(src, ParamFeatures) and src.d == 1
    phi = src.features(population.specs[:3], [b"p1", b"p2"])
    assert phi.shape == (3, 2, 1)
    np.testing.assert_allclose(phi[:, 0, 0], np.log([s.params_b for s in population.specs[:3]]))
    np.testing.assert_array_equal(phi[:, 0], phi[:, 1])   # prompt-independent
