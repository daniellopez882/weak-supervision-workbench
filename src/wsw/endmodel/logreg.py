"""Multinomial logistic regression trained on probabilistic (soft) labels.

This is the "noise-aware" loss of data programming: the cross-entropy against the label
model's posterior ``q_i`` rather than a hard label, i.e.
``sum_i w_i * (-sum_c q_ic log softmax(x_i W + b)_c) + ||W||^2 / (2C)``.
With one-hot ``q`` and unit weights it is exactly scikit-learn's ``LogisticRegression``
objective (intercept not penalised), which the tests use as a parity reference. Like
scikit-learn we minimise it divided by ``sum_i w_i`` so that the gradient tolerance does
not depend on the number of rows.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize

from wsw.labelmodels.base import softmax_rows


class SoftLabelLogisticRegression:
    def __init__(self, *, C: float = 1.0, max_iter: int = 500, tol: float = 1e-5) -> None:
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.coef_: np.ndarray | None = None
        self.intercept_: np.ndarray | None = None
        self.n_iter_ = 0

    def fit(
        self,
        X: sp.csr_matrix | np.ndarray,
        Q: np.ndarray,
        sample_weight: np.ndarray | None = None,
        *,
        warm_start: SoftLabelLogisticRegression | None = None,
    ) -> Self:
        n, d = X.shape
        if Q.ndim != 2 or Q.shape[0] != n:
            raise ValueError("Q must be (n, k) soft labels aligned with X")
        if not np.allclose(Q.sum(axis=1), 1.0, atol=1e-6):
            raise ValueError("each row of Q must sum to one")
        k = Q.shape[1]
        w = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        scale = 1.0 / w.sum()
        inv_c = 1.0 / self.C

        # A CSR copy of X^T makes the gradient product ~2.5x faster than going through CSC.
        XT = sp.csr_matrix(X.T) if sp.issparse(X) else X.T

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            W = theta[: d * k].reshape(d, k)
            b = theta[d * k :]
            Z = X @ W + b
            Z -= Z.max(axis=1, keepdims=True)
            log_norm = np.log(np.exp(Z).sum(axis=1, keepdims=True))
            log_p = Z - log_norm
            loss = scale * float(-(w[:, None] * Q * log_p).sum() + 0.5 * inv_c * (W * W).sum())
            G = scale * w[:, None] * (np.exp(log_p) - Q)
            grad_W = XT @ G + scale * inv_c * W
            return loss, np.concatenate([np.asarray(grad_W).ravel(), G.sum(axis=0)])

        if warm_start is not None and warm_start.coef_ is not None:
            assert warm_start.intercept_ is not None
            theta0 = np.concatenate([warm_start.coef_.ravel(), warm_start.intercept_])
        else:
            theta0 = np.zeros(d * k + k)
        result = minimize(
            objective,
            theta0,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "gtol": self.tol, "ftol": 1e-12},
        )
        self.coef_ = result.x[: d * k].reshape(d, k)
        self.intercept_ = result.x[d * k :]
        self.n_iter_ = int(result.nit)
        return self

    def predict_proba(self, X: sp.csr_matrix | np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None:
            raise RuntimeError("model is not fitted")
        return softmax_rows(np.asarray(X @ self.coef_) + self.intercept_)

    def predict(self, X: sp.csr_matrix | np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)
