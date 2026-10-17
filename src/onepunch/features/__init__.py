"""Log-prob backends, chunk features, cache and the model-major feature extractor."""

from onepunch.features.cache import LogprobCache
from onepunch.features.chunking import chunk_bounds, chunk_features, chunk_features_torch, chunk_weights
from onepunch.features.extractor import FeatureExtractor
from onepunch.features.hf_backend import HFBackend, sequence_logprobs
from onepunch.features.model_pool import ModelPool
from onepunch.features.params import ParamFeatures
from onepunch.features.spans import SpanTokenizer

DEFAULT_FEATURE_SOURCE = "onepunch.features.extractor.FeatureExtractor"


def build_feature_source(cfg, persist: bool | None = None, share_with=None):
    """The FeatureSource named by `cfg.features._target_` (default: chunked log-probs).

    Every feature-source class implements `from_cfg(cfg, persist=None, share_with=None)` and
    `features(models, prompts) -> [N, P, d]`, plus an attribute `d`.
    """
    from hydra.utils import get_class

    cls = get_class(cfg.features.get("_target_", DEFAULT_FEATURE_SOURCE))
    return cls.from_cfg(cfg, persist=persist, share_with=share_with)

__all__ = [
    "FeatureExtractor", "HFBackend", "LogprobCache", "ModelPool", "ParamFeatures", "SpanTokenizer",
    "build_feature_source",
    "chunk_bounds", "chunk_features", "chunk_features_torch", "chunk_weights", "sequence_logprobs",
]
