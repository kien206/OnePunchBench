"""HF backend on SmolLM2-135M (downloads the model on first run; needs a GPU)."""

import gc

import numpy as np
import pytest
import torch

pytestmark = [pytest.mark.slow, pytest.mark.gpu]

MODEL = "HuggingFaceTB/SmolLM2-135M"
PREFIX = b"\n"
TEXT = (
    "The quick brown fox jumps over the lazy dog. Tiếng Việt có dấu, 日本語も、emoji 😀.\n"
    "Photosynthesis converts light energy into chemical energy stored in glucose. "
)
PROMPTS = [(TEXT * k)[: 150 * k + 7].encode("utf-8", "ignore").decode("utf-8", "ignore").encode() for k in (1, 2, 3, 5, 8, 4)]


@pytest.fixture(scope="module", autouse=True)
def _need_gpu():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


def make_backend(dtype, batch_size, model=MODEL):
    from onepunch.features.hf_backend import HFBackend
    from onepunch.types import ModelSpec

    return HFBackend(ModelSpec(model, dtype=dtype), device="cuda", batch_size=batch_size)


def reference_total(backend, p):
    """Plain single-sequence forward, no padding/mask, full float32 log_softmax."""
    tok = backend.tokenizer
    ids = [tok.bos_token_id] + tok(PREFIX.decode(), add_special_tokens=False)["input_ids"]
    s = len(ids)
    ids += tok(p.decode(), add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids], device="cuda")
    with torch.no_grad():
        lp = torch.log_softmax(backend.model(x).logits[0].float(), dim=-1)
    return lp[torch.arange(s - 1, len(ids) - 1), x[0, s:]].double().sum().item(), len(ids) - s


@pytest.mark.parametrize("dtype,tol", [("float32", 5e-3), ("bfloat16", 2.0)])
def test_logprobs_valid_and_match_reference(dtype, tol):
    be = make_backend(dtype, 4)
    try:
        res = be.token_logprobs(PROMPTS, PREFIX)
        for p, tl in zip(PROMPTS, res):
            assert np.isfinite(tl.logprobs).all() and (tl.logprobs <= 0).all()
            assert tl.n_bytes == len(p) and tl.spans[-1, 1] == len(p)
            total, n = reference_total(be, p)
            assert len(tl.logprobs) == n
            assert abs(float(tl.logprobs.astype(np.float64).sum()) - total) < tol, (dtype, total)
    finally:
        be.close()
        gc.collect()


@pytest.mark.parametrize("dtype,tol", [("float32", 1e-3), ("bfloat16", 0.5)])
def test_batch_size_invariance_and_repeatability(dtype, tol, capsys):
    be = make_backend(dtype, 1)
    try:
        r1 = be.token_logprobs(PROMPTS, PREFIX)
        be.batch_size = 4
        r4 = be.token_logprobs(PROMPTS, PREFIX)
        r4b = be.token_logprobs(PROMPTS, PREFIX)
        be.batch_size = 1
        r1b = be.token_logprobs(PROMPTS, PREFIX)
        tok_diff = max(np.abs(a.logprobs - b.logprobs).max() for a, b in zip(r1, r4))
        tot_diff = max(abs(float(a.logprobs.sum()) - float(b.logprobs.sum())) for a, b in zip(r1, r4))
        with capsys.disabled():
            print(f"\n[{dtype}] batch 1 vs 4: max |dlogprob| per token = {tok_diff:.3e}, per prompt sum = {tot_diff:.3e}")
        assert tok_diff < tol
        for a, b in zip(r4, r4b):
            np.testing.assert_array_equal(a.logprobs, b.logprobs)
        for a, b in zip(r1, r1b):
            np.testing.assert_array_equal(a.logprobs, b.logprobs)
        for a, b in zip(r1, r4):
            np.testing.assert_array_equal(a.spans, b.spans)
    finally:
        be.close()


def test_sequence_logprobs_inputs_embeds_matches_ids():
    from onepunch.features.hf_backend import sequence_logprobs

    be = make_backend("float32", 4)
    try:
        ids = torch.tensor([[0, 10, 200, 3000, 41]], device="cuda")
        a = sequence_logprobs(be.model, ids)
        emb = be.model.get_input_embeddings()(ids).detach().requires_grad_(True)
        b = sequence_logprobs(be.model, ids, inputs_embeds=emb)
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
        b.sum().backward()
        assert emb.grad is not None and torch.isfinite(emb.grad).all()
    finally:
        be.close()


def test_empty_context_raises_without_bos():
    from onepunch.features.hf_backend import HFBackend

    be = HFBackend.__new__(HFBackend)
    class T:  # tokenizer without BOS
        bos_token_id = None
    be._tokenizer = T()
    with pytest.raises(ValueError):
        be.context_ids(b"")


def test_sentencepiece_model_matches_reference_and_chunk_sum():
    """TinyLlama (Llama-2 tokenizer: dummy prefix, byte fallback)."""
    from onepunch.features.chunking import chunk_features, chunk_sizes

    be = make_backend("float32", 4, "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T")
    try:
        ps = PROMPTS[:3]
        for p, tl in zip(ps, be.token_logprobs(ps, PREFIX)):
            total, n = reference_total(be, p)
            assert len(tl.logprobs) == n and abs(float(tl.logprobs.sum()) - total) < 5e-3
            assert tl.spans[0, 0] == 0 and tl.spans[0, 1] > 0
            # the dummy-prefix space belongs to no chunk, but it is not a separate token here
            f = chunk_features(tl, 8)
            assert abs((f * chunk_sizes(tl.n_bytes, 8)).sum() - tl.logprobs.sum()) < 1e-3
    finally:
        be.close()


def test_chunk_sum_identity_from_cfg(tmp_path):
    """FeatureExtractor.from_cfg end to end: sum_j |c_j| f_j = sum of token log-probs."""
    from hydra import compose, initialize_config_dir

    from onepunch.features import FeatureExtractor
    from onepunch.features.chunking import chunk_sizes
    from onepunch.types import ModelSpec

    from omegaconf import OmegaConf

    # compose only this package's groups, so the test does not depend on the other config groups
    with initialize_config_dir(str(__import__("pathlib").Path(__file__).parents[1] / "configs"), version_base="1.3"):
        cfg = compose(config_name=None, overrides=["+backend=hf", "+features=chunk16", "backend.batch_size=3",
                                                   "backend.pool_ram_gb=1"])
    OmegaConf.set_struct(cfg, False)
    cfg.paths = {"cache": str(tmp_path)}
    ex = FeatureExtractor.from_cfg(cfg)
    spec = ModelSpec(MODEL, dtype="float32")
    try:
        prompts = [p for p in PROMPTS if len(p) >= 16 * 32]
        assert prompts
        phi = ex.features([spec], prompts)
        tls = ex.token_logprobs([spec], prompts)[0]
        for f, tl in zip(phi[0], tls):
            np.testing.assert_allclose((f * chunk_sizes(tl.n_bytes, 16)).sum(), tl.logprobs.sum(), rtol=1e-5)
        assert list(tmp_path.rglob("*.npz"))
    finally:
        ex.pool.clear()
