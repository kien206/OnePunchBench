"""Episodic outer loss: ridge fitted on the inner set, scored on the outer set.

Normalization lives here (not in the readout): features and scores are centered with the
INNER means and scores are divided by sigma, so every readout sees the same inputs.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from onepunch.readout.base import Readout
from onepunch.types import Episode


def standardize_scales(S_train: np.ndarray, standardize: bool = True, eps: float = 1e-8) -> np.ndarray:
    """sigma_k = std of benchmark k over all training models (ones if standardize=False)."""
    S = np.asarray(S_train, dtype=np.float64)
    if not standardize:
        return np.ones(S.shape[1])
    sig = S.std(axis=0)
    return np.where(sig < eps, 1.0, sig)


def _t(x) -> torch.Tensor:
    return torch.as_tensor(x).to(torch.float64) if not isinstance(x, torch.Tensor) else x.to(torch.float64)


def fit_predict(Phi_in: torch.Tensor, S_in: torch.Tensor, Phi_out: torch.Tensor, readout: Readout,
                sigma=None, center: bool = True) -> torch.Tensor:
    """Fit on (Phi_in, S_in), predict Phi_out; return predictions in RAW score units.

    Phi_in [..., N_in, d] (optional leading prompt dim), S_in [N_in, K], Phi_out [..., N_out, d].
    Centering with inner means and sigma scaling are applied internally and undone on output.
    """
    Phi_in, S_in, Phi_out = _t(Phi_in), _t(S_in), _t(Phi_out)
    if Phi_in.ndim == 3 and not getattr(readout, "supports_batch", False):
        return torch.stack([fit_predict(Phi_in[i], S_in, Phi_out[i], readout, sigma, center)
                            for i in range(Phi_in.shape[0])])
    sig = torch.ones(S_in.shape[1], dtype=torch.float64) if sigma is None else _t(sigma)
    if center:
        mu_f, mu_s = Phi_in.mean(-2, keepdim=True), S_in.mean(0)
    else:
        mu_f, mu_s = torch.zeros_like(Phi_in[..., :1, :]), torch.zeros_like(S_in[0])
    fitted = readout.fit(Phi_in - mu_f, (S_in - mu_s) / sig)
    return mu_s + fitted.predict(Phi_out - mu_f) * sig


def _split_loss(Phi_in, S_in, Phi_out, S_out, readout, sigma, center) -> torch.Tensor:
    """||(pred - S_out)/sigma||_F^2 / (N_out K); [...,] over the leading prompt dim."""
    S_out = _t(S_out)
    pred = fit_predict(Phi_in, S_in, Phi_out, readout, sigma, center)
    sig = 1.0 if sigma is None else _t(sigma)
    return (((pred - S_out) / sig) ** 2).mean(dim=(-2, -1))


def episode_losses(Phi, S, episodes: Sequence[Episode], readout: Readout, sigma=None,
                   center: bool = True) -> torch.Tensor:
    """Per-episode outer losses l_e.

    Phi: [N, d] -> returns [E];  Phi: [N, P, d] (P prompts) -> returns [E, P].
    Gradients flow through centering and the readout fit back to Phi.
    """
    Phi, S = _t(Phi), _t(S)
    if Phi.ndim == 3:
        Phi = Phi.permute(1, 0, 2)  # [P, N, d]
    out = []
    for ep in episodes:
        i, o = torch.as_tensor(ep.in_idx), torch.as_tensor(ep.out_idx)
        out.append(_split_loss(Phi[..., i, :], S[i], Phi[..., o, :], S[o], readout, sigma, center))
    return torch.stack(out)


def episode_loss(Phi, S, episodes: Sequence[Episode], readout: Readout, sigma=None,
                 center: bool = True) -> torch.Tensor:
    """Mean over episodes of l_e: scalar for Phi [N, d], [P] for Phi [N, P, d]."""
    return episode_losses(Phi, S, episodes, readout, sigma, center).mean(0)


def validation_loss(Phi_train, S_train, Phi_val, S_val, readout: Readout, sigma=None,
                    center: bool = True) -> torch.Tensor:
    """Fit on all training models, score on the validation models (same normalization).

    Phi [N, d] -> scalar; Phi [N, P, d] -> [P].
    """
    Phi_train, Phi_val = _t(Phi_train), _t(Phi_val)
    if Phi_train.ndim == 3:
        Phi_train, Phi_val = Phi_train.permute(1, 0, 2), Phi_val.permute(1, 0, 2)
    return _split_loss(Phi_train, _t(S_train), Phi_val, S_val, readout, sigma, center)
