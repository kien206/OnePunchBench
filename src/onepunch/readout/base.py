"""Readout interface: maps features Phi [N, d] to scores S [N, K].

FROZEN contract. Ridge is the default (closed form, differentiable, exact);
an MLP or other readout drops in by subclassing `Readout`.

Inputs are already centered (and scores standardized) by the caller
(episodes/loss.py, eval/lofo.py), so every readout sees the same normalization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class Readout(ABC):
    """Configured (unfitted) readout. Hyperparameters live on the instance.

    `fit` must NOT mutate `self`: it returns a new fitted object, so one
    configured readout can be fitted on many episodes.
    """

    #: True if `fit` is an explicit differentiable function of Phi (autograd
    #: flows from predictions back to the inner features). Required by the
    #: bilevel surrogate hypergradient.
    closed_form: bool = False

    @abstractmethod
    def fit(self, Phi: torch.Tensor, S: torch.Tensor) -> "FittedReadout":
        """Fit on Phi [N, d], S [N, K] (float64, centered)."""


class FittedReadout(ABC):
    @abstractmethod
    def predict(self, Phi: torch.Tensor) -> torch.Tensor:
        """Phi [M, d] -> predictions [M, K] in the same normalized space."""

    @abstractmethod
    def state_dict(self) -> dict[str, Any]:
        """Tensors/values needed to reproduce predictions at inference (e.g. W)."""
