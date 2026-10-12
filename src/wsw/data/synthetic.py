"""Label matrices drawn from the model every label model here assumes.

Each LF fires independently of ``Y`` with probability ``coverage[j]``; when it fires it is
right with probability ``accuracy[j]`` and otherwise names one of the other classes uniformly.
LFs are conditionally independent given ``Y``. Because the true parameters are known, these
matrices are what the tests and the parameter-recovery benchmark score estimates against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wsw.lf import ABSTAIN, LABEL_DTYPE


@dataclass(frozen=True)
class SyntheticLabels:
    L: np.ndarray
    y: np.ndarray
    accuracy: np.ndarray
    coverage: np.ndarray
    balance: np.ndarray

    @property
    def k(self) -> int:
        return int(self.balance.shape[0])


def sample_label_matrix(
    n: int,
    accuracy: np.ndarray,
    coverage: np.ndarray,
    balance: np.ndarray,
    seed: int,
) -> SyntheticLabels:
    accuracy = np.asarray(accuracy, dtype=np.float64)
    coverage = np.asarray(coverage, dtype=np.float64)
    balance = np.asarray(balance, dtype=np.float64)
    if accuracy.shape != coverage.shape or accuracy.ndim != 1:
        raise ValueError("accuracy and coverage must be 1-D arrays of the same length")
    if not np.isclose(balance.sum(), 1.0) or (balance <= 0).any():
        raise ValueError("balance must be a positive probability vector")
    k, m = balance.shape[0], accuracy.shape[0]
    rng = np.random.default_rng(seed)
    y = rng.choice(k, size=n, p=balance)
    fires = rng.random((n, m)) < coverage
    correct = rng.random((n, m)) < accuracy
    wrong = (y[:, None] + rng.integers(1, k, size=(n, m))) % k
    L = np.where(fires, np.where(correct, y[:, None], wrong), ABSTAIN).astype(LABEL_DTYPE)
    return SyntheticLabels(L, y.astype(np.int64), accuracy, coverage, balance)


def random_lf_parameters(
    m: int, seed: int, *, accuracy: tuple[float, float], coverage: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.uniform(*accuracy, size=m), rng.uniform(*coverage, size=m)
