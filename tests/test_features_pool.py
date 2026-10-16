"""ModelPool with a real small model: pooled and direct loads must agree (needs a GPU)."""

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

from onepunch.features.hf_backend import HFBackend
from onepunch.features.model_pool import ModelPool
from onepunch.types import ModelSpec

SPEC = ModelSpec("HuggingFaceTB/SmolLM2-135M", dtype="float32")
PROMPTS = [b"The pool keeps weights in pinned memory. " * 4, "Xin chào, 日本語, 😀 ".encode() * 6]


@pytest.fixture(autouse=True)
def _need_gpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


@pytest.mark.parametrize("capacity_gb", [0.0, 2.0])
def test_pool_matches_direct_and_moves_weights(capacity_gb):
    direct = HFBackend(SPEC, batch_size=2)
    ref = direct.token_logprobs(PROMPTS, b"\n")
    direct.close()

    pool = ModelPool(capacity_gb, "cuda", batch_size=2)
    try:
        for _ in range(2):  # second round hits the pinned-RAM copy when capacity > 0
            be = pool.acquire(SPEC)
            assert next(be.model.parameters()).is_cuda
            got = be.token_logprobs(PROMPTS, b"\n")
            for a, b in zip(ref, got):
                np.testing.assert_allclose(a.logprobs, b.logprobs, atol=1e-4)
            model = be.model
            pool.release(SPEC)
            if capacity_gb > 0:
                assert not next(model.parameters()).is_cuda and pool.host_bytes > 0
            else:
                assert pool.host_bytes == 0
    finally:
        pool.clear()
    assert pool.host_bytes == 0
