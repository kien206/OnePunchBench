"""Leave-one-family-out evaluation and the stored inference artifact (formulation: Inference)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from onepunch.episodes.loss import fit_predict, standardize_scales
from onepunch.readout.base import Readout


def lofo_predict(Phi: np.ndarray, S: np.ndarray, families: Sequence, readout: Readout,
                 standardize: bool = True) -> np.ndarray:
    """Predict every model's scores from a readout fitted on all OTHER families.

    Each fold centers with its own training means and uses its own sigma; predictions are in raw
    score space, clipped to [0,1]. Phi may carry extra non-prompt columns (e.g. log params).
    """
    Phi = np.asarray(Phi, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    fam = np.asarray(families)
    preds = np.empty_like(S)
    for f in sorted(set(fam.tolist())):
        te = fam == f
        tr = ~te
        sigma = standardize_scales(S[tr], standardize)
        with torch.no_grad():
            p = fit_predict(torch.as_tensor(Phi[tr]), torch.as_tensor(S[tr]),
                            torch.as_tensor(Phi[te]), readout, sigma)
        preds[te] = np.clip(p.numpy(), 0.0, 1.0)
    return preds


@dataclass
class FittedPredictor:
    """s_hat = clip(s_bar + D_sigma W^T (f - f_bar)). `meta` may hold the prompt, d, p0, ..."""

    W: np.ndarray       # [d, K]
    f_bar: np.ndarray   # [d]
    s_bar: np.ndarray   # [K]
    sigma: np.ndarray   # [K]
    meta: dict[str, Any] = field(default_factory=dict)

    def predict(self, Phi_new: np.ndarray, clip: bool = True) -> np.ndarray:
        Phi_new = np.asarray(Phi_new, dtype=np.float64)
        out = self.s_bar + ((Phi_new - self.f_bar) @ self.W) * self.sigma
        return np.clip(out, 0.0, 1.0) if clip else out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            np.savez(fh, W=self.W, f_bar=self.f_bar, s_bar=self.s_bar, sigma=self.sigma,
                     meta=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path: str | Path) -> "FittedPredictor":
        z = np.load(path, allow_pickle=False)
        return cls(z["W"], z["f_bar"], z["s_bar"], z["sigma"], json.loads(str(z["meta"])))


def fit_predictor(Phi: np.ndarray, S: np.ndarray, readout: Readout, standardize: bool = True,
                  meta: dict[str, Any] | None = None) -> FittedPredictor:
    """Fit on all rows; store W, feature mean, score mean, sigma (needs a linear readout with `W`)."""
    Phi = np.asarray(Phi, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    sigma = standardize_scales(S, standardize)
    f_bar, s_bar = Phi.mean(0), S.mean(0)
    fitted = readout.fit(torch.as_tensor(Phi - f_bar), torch.as_tensor((S - s_bar) / sigma))
    state = fitted.state_dict()
    if "W" not in state:
        raise ValueError("fit_predictor needs a linear readout whose state_dict has 'W'")
    return FittedPredictor(np.asarray(state["W"], dtype=np.float64), f_bar, s_bar, sigma, dict(meta or {}))
