"""Closed-form ridge readout, differentiable in Phi (formulation: Eq. inner)."""

from __future__ import annotations

from typing import Any, Literal

import torch

from onepunch.readout.base import FittedReadout, Readout


class FittedRidge(FittedReadout):
    """Linear predictor Phi -> Phi @ W (inputs already centered by the caller)."""

    def __init__(self, W: torch.Tensor):
        self.W = W

    def predict(self, Phi: torch.Tensor) -> torch.Tensor:
        return Phi.to(self.W.dtype) @ self.W

    def state_dict(self) -> dict[str, Any]:
        return {"W": self.W.detach().cpu()}


class RidgeReadout(Readout):
    """W = (PhiᵀPhi + λ I)⁻¹ PhiᵀS, computed in whichever form inverts the smaller matrix.

    λ = lambda0 * N if `scale_with_n` (so lambda0 means the same for any N).
    Also accepts a leading batch dim: Phi [B, N, d] with S [N, K] or [B, N, K]
    (used to score many candidate prompts at once).
    """

    closed_form = True
    supports_batch = True

    def __init__(self, lambda0: float, scale_with_n: bool = True,
                 form: Literal["auto", "primal", "dual"] = "auto"):
        if form not in ("auto", "primal", "dual"):
            raise ValueError(f"form must be auto|primal|dual, got {form!r}")
        self.lambda0 = float(lambda0)
        self.scale_with_n = scale_with_n
        self.form = form

    def fit(self, Phi: torch.Tensor, S: torch.Tensor) -> FittedRidge:
        Phi = Phi.to(torch.float64)
        S = S.to(torch.float64)
        n, d = Phi.shape[-2], Phi.shape[-1]
        if Phi.ndim == 3 and S.ndim == 2:
            S = S.expand(Phi.shape[0], *S.shape)
        lam = self.lambda0 * n if self.scale_with_n else self.lambda0
        form = self.form if self.form != "auto" else ("primal" if d <= n else "dual")
        Pt = Phi.transpose(-1, -2)
        if form == "primal":
            A = Pt @ Phi + lam * torch.eye(d, dtype=Phi.dtype, device=Phi.device)
            W = torch.linalg.solve(A, Pt @ S)
        else:
            G = Phi @ Pt + lam * torch.eye(n, dtype=Phi.dtype, device=Phi.device)
            W = Pt @ torch.linalg.solve(G, S)
        return FittedRidge(W)
