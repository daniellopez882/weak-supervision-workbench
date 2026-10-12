"""Class-balance estimation from third-order LF agreements (MeTaL's tensor approach).

Second moments of conditionally independent LFs do not identify ``P(Y)`` (for two
classes any balance can be traded against the LF parameters), but the three-way overlap
tensor does. With abstain kept as an outcome, for distinct LFs ``i, j, l``::

    P(lambda_i = a, lambda_j = b, lambda_l = c) = sum_y P(y) C_i[a, y] C_j[b, y] C_l[c, y]

Writing ``Q_i = C_i P(y)^(1/3)`` makes the right-hand side a symmetric CP decomposition
``sum_y Q[:, y] (x) Q[:, y] (x) Q[:, y]`` over the stacked outcome index. We fit ``Q`` to
the empirical tensor on entries with three distinct LFs, then read the balance off the
column sums, because every column of ``C_i`` sums to one: ``sum_a Q_i[a, y] = P(y)^(1/3)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

from wsw.lf import ABSTAIN, check_label_matrix


@dataclass(frozen=True)
class ClassBalanceEstimate:
    balance: np.ndarray
    lf_indices: np.ndarray
    loss: float
    converged: bool


def _triplet_tensor(outcomes: np.ndarray, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Empirical overlap tensor over the stacked (LF, outcome) index, plus its mask."""
    patterns, counts = np.unique(outcomes, axis=0, return_counts=True)
    n_lfs = outcomes.shape[1]
    D = n_lfs * width
    X = np.zeros((patterns.shape[0], D))
    X[np.arange(patterns.shape[0])[:, None], np.arange(n_lfs) * width + patterns] = 1.0
    w = counts / counts.sum()
    pair = np.einsum("ua,ub->uab", X * w[:, None], X).reshape(patterns.shape[0], D * D)
    O = (pair.T @ X).reshape(D, D, D)
    lf_of = np.repeat(np.arange(n_lfs), width)
    a, b, c = np.meshgrid(lf_of, lf_of, lf_of, indexing="ij")
    mask = ((a != b) & (a != c) & (b != c)).astype(np.float64)
    return O, mask


def estimate_class_balance(
    L: np.ndarray,
    k: int,
    *,
    max_lfs: int = 10,
    prec_init: float = 0.7,
    max_iter: int = 500,
) -> ClassBalanceEstimate:
    """Estimate ``P(Y)`` without labels from the ``max_lfs`` highest-coverage LFs."""
    L = check_label_matrix(L, k)
    coverage = (L != ABSTAIN).mean(axis=0)
    order = np.argsort(-coverage, kind="stable")
    chosen = np.sort(order[: min(max_lfs, L.shape[1])])
    if chosen.size < 3:
        raise ValueError("class-balance estimation needs at least three labelling functions")
    width = k + 1
    outcomes = np.where(L[:, chosen] == ABSTAIN, k, L[:, chosen])
    O, mask = _triplet_tensor(outcomes, width)
    n_lfs = chosen.size
    D = n_lfs * width

    # Start from "each LF votes its own class with precision prec_init" under a uniform prior.
    C0 = np.zeros((n_lfs, width, k))
    for i, cov in enumerate(coverage[chosen]):
        off = cov * (1.0 - prec_init) / max(k - 1, 1)
        C0[i, :k, :] = off
        C0[i, np.arange(k), np.arange(k)] = cov * prec_init
        C0[i, k, :] = 1.0 - cov
    Q0 = (C0 * (1.0 / k) ** (1.0 / 3.0)).reshape(D, k)

    def objective(flat: np.ndarray) -> tuple[float, np.ndarray]:
        Q = flat.reshape(D, k)
        T = np.einsum("ay,by,cy->abc", Q, Q, Q, optimize=True)
        R = mask * (T - O)
        loss = float((R * R).sum())
        # R is symmetric in its three indices, so the three chain-rule terms coincide.
        grad = 6.0 * np.einsum("abc,by,cy->ay", R, Q, Q, optimize=True)
        return loss, grad.ravel()

    result = minimize(
        objective,
        Q0.ravel(),
        jac=True,
        method="L-BFGS-B",
        bounds=[(0.0, 1.0)] * (D * k),
        options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-12},
    )
    Q = result.x.reshape(n_lfs, width, k)

    # Columns of Q are only identified up to a permutation; pick the one under which LF
    # votes agree most with the class they are attributed to.
    agreement = Q[:, :k, :].sum(axis=0)
    rows, cols = linear_sum_assignment(-agreement)
    perm = cols[np.argsort(rows)]
    Q = Q[:, :, perm]

    cube_roots = Q.sum(axis=1).mean(axis=0)
    balance = np.clip(cube_roots, 1e-12, None) ** 3
    return ClassBalanceEstimate(
        balance=balance / balance.sum(),
        lf_indices=chosen,
        loss=float(result.fun),
        converged=bool(result.success),
    )


def balance_from_labels(y: np.ndarray, k: int, *, floor: float = 1e-3) -> np.ndarray:
    """Class frequencies of a labelled sample (WRENCH takes them from the validation split).

    The floor keeps a class that is missing from a small sample at a non-zero prior, which
    the label models need for ``log P(y)``.
    """
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=k).astype(np.float64)
    if counts.shape != (k,) or counts.sum() == 0:
        raise ValueError(f"need at least one label in [0, {k})")
    p = np.maximum(counts / counts.sum(), floor)
    return p / p.sum()
