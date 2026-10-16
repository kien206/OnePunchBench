"""Prompts must be NFC: Qwen and Pythia tokenizers NFC-normalize input, so a non-NFC prompt
would be read as different bytes by different models."""

import numpy as np

from onepunch.search import BestOfN, PromptConstraints
from onepunch.types import SearchState


def test_constraints_reject_non_nfc():
    c = PromptConstraints(min_bytes=1, max_bytes=100)
    assert c.check("café".encode()) == (True, "ok")
    assert c.check("café".encode()) == (False, "not_nfc")


def test_random_bytes_prompts_are_nfc():
    c = PromptConstraints(min_bytes=512, max_bytes=512)
    state = SearchState(step=0, prompt=b"", loss=np.nan, best_prompt=b"", best_loss=np.nan,
                        rng=np.random.default_rng(0))
    for p in BestOfN("random_bytes", 512, 50).propose(state):
        assert c.check(p) == (True, "ok")
