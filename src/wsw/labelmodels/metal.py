"""MeTaL label model (Ratner et al., AAAI 2019): matrix completion on LF overlaps.

With one-hot "augmented" votes ``A`` (abstains are all-zero rows) and
``O = A^T A / n``, conditional independence of the LFs given ``Y`` implies that every
off-diagonal ``k x k`` block of ``O`` equals ``mu_i P mu_j^T`` with
``mu[i*k + l, y] = P(lambda_i = l | Y = y)`` and ``P = diag(P(Y))``. We fit ``mu`` to
those blocks plus the per-LF vote marginals ``mu P 1 = diag(O)``, which is the objective
of Snorkel's ``LabelModel``, solved here with L-BFGS-B and box constraints instead of SGD.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal, Self

import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

from wsw.labelmodels.base import LabelModel, resolve_class_balance, softmax_rows
from wsw.labelmodels.class_balance import estimate_class_balance
from wsw.lf import check_label_matrix, one_hot

BalanceSpec = np.ndarray | Literal["uniform", "estimate"]


def augmented(L: np.ndarray, k: int) -> np.ndarray:
    """``(n, m*k)`` indicator of ``lambda_j == l``; abstains contribute no 1."""
    return one_hot(L, k, abstain_column=False).reshape(L.shape[0], -1)


def class_balance_for(spec: BalanceSpec, L: np.ndarray, k: int) -> tuple[np.ndarray, str]:
    if isinstance(spec, str):
        if spec == "uniform":
            return resolve_class_balance(None, k), "uniform"
        if spec == "estimate":
            return estimate_class_balance(L, k).balance, "estimated"
        raise ValueError(f"unknown class balance option {spec!r}")
    return resolve_class_balance(spec, k), "given"


class MetalObjective:
    """The matrix-completion loss, exposed so other fits (e.g. Snorkel's) can be scored on it."""

    def __init__(
        self, L: np.ndarray, k: int, p: np.ndarray, *, prec_init: float = 0.7, l2: float = 0.0
    ) -> None:
        n, m = L.shape
        self.k, self.p, self.l2 = k, p, l2
        self.d = m * k
        A = augmented(L, k)
        self.O = A.T @ A / n
        self.marginals = np.diag(self.O).copy()
        self.mask = np.ones((self.d, self.d))
        for j in range(m):
            self.mask[j * k : (j + 1) * k, j * k : (j + 1) * k] = 0.0
        # Snorkel's initialisation: P(lambda_j = y | Y = y) = P(lambda_j = y) * prec / P(y).
        rows = np.arange(self.d)
        self.mu_init = np.zeros((self.d, k))
        self.mu_init[rows, rows % k] = np.clip(self.marginals * prec_init / p[rows % k], 0.0, 1.0)

    def unflatten(self, flat: np.ndarray) -> np.ndarray:
        return flat.reshape(self.d, self.k)

    def value_and_grad(self, flat: np.ndarray) -> tuple[float, np.ndarray]:
        mu = self.unflatten(flat)
        mu_p = mu * self.p
        R = self.mask * (mu_p @ mu.T - self.O)
        r_marg = mu @ self.p - self.marginals
        diff = mu - self.mu_init
        loss = float((R * R).sum() + r_marg @ r_marg + self.l2 * (diff * diff).sum())
        grad = 4.0 * R @ mu_p + 2.0 * np.outer(r_marg, self.p) + 2.0 * self.l2 * diff
        return loss, grad.ravel()

    def value(self, mu: np.ndarray) -> float:
        return self.value_and_grad(np.asarray(mu, dtype=np.float64).ravel())[0]


class MeTaL(LabelModel):
    name = "metal"

    def __init__(
        self,
        *,
        class_balance: BalanceSpec = "estimate",
        prec_init: float = 0.7,
        l2: float = 0.01,
        max_iter: int = 1000,
    ) -> None:
        super().__init__()
        self.class_balance = class_balance
        self.prec_init = prec_init
        self.l2 = l2
        self.max_iter = max_iter
        self.mu_: np.ndarray | None = None
        self.class_balance_: np.ndarray | None = None
        self.class_balance_source_ = ""
        self.loss_ = float("nan")

    def fit(self, L: np.ndarray, k: int) -> Self:
        L = check_label_matrix(L, k)
        self.k = k
        p, self.class_balance_source_ = class_balance_for(self.class_balance, L, k)
        objective = MetalObjective(L, k, p, prec_init=self.prec_init, l2=self.l2)
        # Same floor as Snorkel so that log(mu) stays finite at prediction time.
        eps = min(0.01, 10.0 ** -np.ceil(np.log10(L.shape[0])))
        size = objective.mu_init.size
        result = minimize(
            objective.value_and_grad,
            np.clip(objective.mu_init, eps, 1.0 - eps).ravel(),
            jac=True,
            method="L-BFGS-B",
            bounds=[(eps, 1.0 - eps)] * size,
            options={"maxiter": self.max_iter, "ftol": 1e-14, "gtol": 1e-10},
        )
        self.loss_ = float(result.fun)
        self.mu_ = _break_permutation_symmetry(objective.unflatten(result.x), p, L.shape[1], k)
        self.class_balance_ = p
        return self

    def lf_accuracy(self) -> np.ndarray:
        assert self.k is not None and self.mu_ is not None and self.class_balance_ is not None
        k, p = self.k, self.class_balance_
        mu = self.mu_.reshape(-1, k, k)  # (LF, vote, true class)
        correct = (mu[:, np.arange(k), np.arange(k)] * p).sum(axis=1)
        votes = (mu.sum(axis=1) * p).sum(axis=1)
        return np.divide(correct, votes, out=np.full_like(correct, np.nan), where=votes > 0)

    def _proba(self, L: np.ndarray) -> np.ndarray:
        assert self.k is not None and self.mu_ is not None and self.class_balance_ is not None
        if L.shape[1] * self.k != self.mu_.shape[0]:
            raise ValueError("label matrix has a different number of LFs than at fit time")
        # Abstains add nothing to any class, i.e. they are treated as uninformative.
        logits = augmented(L, self.k) @ np.log(self.mu_) + np.log(self.class_balance_)
        return softmax_rows(logits)


def _break_permutation_symmetry(mu: np.ndarray, p: np.ndarray, m: int, k: int) -> np.ndarray:
    """Columns of ``mu`` with equal prior can swap without changing the loss; choose the
    assignment that maximises total agreement, as Snorkel does with the Munkres solver."""
    probs_sum = mu.reshape(m, k, k).sum(axis=0) * p
    Z = np.zeros((k, k))
    groups: defaultdict[float, list[int]] = defaultdict(list)
    for y, prior in enumerate(p):
        groups[round(float(prior), 3)].append(y)
    for group in groups.values():
        sub = probs_sum[np.ix_(group, group)]
        rows, cols = linear_sum_assignment(-sub.T)
        for r, c in zip(rows, cols, strict=True):
            Z[group[r], group[c]] = 1.0
    return mu @ Z
