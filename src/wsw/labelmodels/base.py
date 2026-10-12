"""Shared interface for label models.

A label model turns an ``(n, m)`` label matrix into an ``(n, k)`` matrix of class
posteriors without seeing gold labels (Dawid-Skene can optionally use a few).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np

from wsw.lf import check_label_matrix


class LabelModel(ABC):
    name: str = "label_model"
    #: True when ``fit`` can take gold labels for a subset of rows (used by the active loop).
    accepts_gold: bool = False

    def __init__(self) -> None:
        self.k: int | None = None

    @abstractmethod
    def fit(self, L: np.ndarray, k: int) -> Self: ...

    @abstractmethod
    def _proba(self, L: np.ndarray) -> np.ndarray: ...

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        if self.k is None:
            raise RuntimeError(f"{self.name} must be fitted before predicting")
        return self._proba(check_label_matrix(L, self.k))

    def predict(self, L: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return argmax_random_ties(self.predict_proba(L), rng)

    def lf_accuracy(self) -> np.ndarray | None:
        """Estimated ``P(lambda_j = Y | lambda_j votes)`` per LF, or ``None`` if the model has
        no per-LF parameters (majority vote) or they do not map onto this quantity."""
        return None


def argmax_random_ties(P: np.ndarray, rng: np.random.Generator, atol: float = 1e-12) -> np.ndarray:
    """Row-wise argmax that breaks exact ties uniformly at random (seeded)."""
    is_max = P >= P.max(axis=1, keepdims=True) - atol
    noise = rng.random(P.shape)
    return np.argmax(np.where(is_max, noise, -1.0), axis=1)


def normalise_rows(X: np.ndarray) -> np.ndarray:
    total = X.sum(axis=1, keepdims=True)
    uniform = np.full_like(X, 1.0 / X.shape[1])
    return np.divide(X, total, out=uniform, where=total > 0)


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    expd = np.exp(shifted)
    return expd / expd.sum(axis=1, keepdims=True)


def resolve_class_balance(class_balance: np.ndarray | None, k: int) -> np.ndarray:
    if class_balance is None:
        return np.full(k, 1.0 / k)
    p = np.asarray(class_balance, dtype=np.float64)
    if p.shape != (k,) or (p <= 0).any():
        raise ValueError(f"class balance must be {k} positive numbers, got {class_balance!r}")
    return p / p.sum()
