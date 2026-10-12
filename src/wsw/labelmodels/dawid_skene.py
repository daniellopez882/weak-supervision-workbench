"""Dawid-Skene (1979) expectation-maximisation over per-LF confusion matrices.

Each LF ``j`` has a confusion matrix ``theta[j, y, o] = P(lambda_j = o | Y = y)``. With
``abstain_as_outcome=True`` (default) abstaining is one of the outcomes ``o``, so a
keyword LF that only ever votes one class still carries information through *when* it
fires. With ``False`` abstains are treated as missing at random and skipped.

Rows listed in ``gold_idx`` keep a one-hot posterior throughout EM, which turns the model
into the semi-supervised variant the active-learning loop uses.
"""

from __future__ import annotations

from typing import Self

import numpy as np
from scipy.special import logsumexp

from wsw.labelmodels.base import LabelModel, normalise_rows, softmax_rows
from wsw.lf import ABSTAIN, check_label_matrix, vote_counts


class DawidSkene(LabelModel):
    name = "dawid_skene"
    accepts_gold = True

    def __init__(
        self,
        *,
        max_iter: int = 200,
        tol: float = 1e-7,
        smoothing: float = 0.01,
        abstain_as_outcome: bool = True,
    ) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.smoothing = smoothing
        self.abstain_as_outcome = abstain_as_outcome
        self.class_prior_: np.ndarray | None = None
        self.log_theta_: np.ndarray | None = None
        self.n_iter_ = 0
        self.log_likelihood_: list[float] = []

    def fit(
        self,
        L: np.ndarray,
        k: int,
        *,
        gold_idx: np.ndarray | None = None,
        gold_y: np.ndarray | None = None,
    ) -> Self:
        L = check_label_matrix(L, k)
        self.k = k
        outcomes = np.where(L == ABSTAIN, k, L)
        gold_idx = np.asarray([] if gold_idx is None else gold_idx, dtype=np.int64)
        gold_y = np.asarray([] if gold_y is None else gold_y, dtype=np.int64)
        if gold_idx.shape != gold_y.shape:
            raise ValueError("gold_idx and gold_y must have the same length")

        # Unlabelled rows with the same vote pattern share a posterior, so EM runs over the
        # distinct patterns weighted by their counts; gold rows are appended one by one.
        unlabelled = np.ones(L.shape[0], dtype=bool)
        unlabelled[gold_idx] = False
        patterns, counts = np.unique(outcomes[unlabelled], axis=0, return_counts=True)
        rows = np.vstack([patterns.reshape(-1, L.shape[1]), outcomes[gold_idx]])
        weights = np.concatenate([counts.astype(np.float64), np.ones(gold_idx.size)])
        clamped = np.arange(patterns.shape[0], rows.shape[0])
        gold_q = np.eye(k)[gold_y]

        # Soft majority vote where LFs vote. Rows nobody votes on start at the class mix of the
        # covered rows rather than uniform: with many low-coverage LFs (SMS) a uniform start
        # lets EM explain "nobody fired" as a class of its own.
        votes = vote_counts(np.where(rows == k, ABSTAIN, rows), k).astype(np.float64)
        q = normalise_rows(votes)
        covered = votes.sum(axis=1) > 0
        if covered.any():
            q[~covered] = (q[covered] * weights[covered, None]).sum(axis=0) / weights[covered].sum()
        q[clamped] = gold_q
        self.log_likelihood_ = []
        prev = -np.inf
        for iteration in range(1, self.max_iter + 1):
            self._m_step(q, rows, weights, k)
            log_joint = self._log_joint(rows)
            # Observed-data log-likelihood: marginal over Y for unlabelled rows, joint for gold.
            row_ll = logsumexp(log_joint, axis=1)
            row_ll[clamped] = log_joint[clamped, gold_y]
            ll = float(weights @ row_ll)
            q = softmax_rows(log_joint)
            q[clamped] = gold_q
            self.log_likelihood_.append(ll)
            self.n_iter_ = iteration
            if abs(ll - prev) <= self.tol * max(1.0, abs(ll)):
                break
            prev = ll
        return self

    def _m_step(self, q: np.ndarray, rows: np.ndarray, weights: np.ndarray, k: int) -> None:
        wq = q * weights[:, None]
        n_eff = wq.sum(axis=0)
        self.class_prior_ = (n_eff + self.smoothing) / (n_eff.sum() + k * self.smoothing)
        theta = np.ones((rows.shape[1], k, k + 1))
        for j in range(rows.shape[1]):
            counts = np.stack(
                [np.bincount(rows[:, j], weights=wq[:, c], minlength=k + 1) for c in range(k)]
            )
            if self.abstain_as_outcome:
                smoothed = counts + self.smoothing
                theta[j] = smoothed / smoothed.sum(axis=1, keepdims=True)
            else:
                smoothed = counts[:, :k] + self.smoothing
                # The abstain column stays at 1 so it adds log(1) = 0 to every class.
                theta[j, :, :k] = smoothed / smoothed.sum(axis=1, keepdims=True)
        self.log_theta_ = np.log(theta)

    def _log_joint(self, outcomes: np.ndarray) -> np.ndarray:
        assert self.class_prior_ is not None and self.log_theta_ is not None
        log_joint = np.tile(np.log(self.class_prior_), (outcomes.shape[0], 1))
        for j in range(outcomes.shape[1]):
            log_joint += self.log_theta_[j][:, outcomes[:, j]].T
        return log_joint

    def _proba(self, L: np.ndarray) -> np.ndarray:
        assert self.k is not None and self.log_theta_ is not None
        if L.shape[1] != self.log_theta_.shape[0]:
            raise ValueError("label matrix has a different number of LFs than at fit time")
        return softmax_rows(self._log_joint(np.where(L == ABSTAIN, self.k, L)))

    def lf_accuracy(self) -> np.ndarray:
        theta, prior = self.confusion_, self.class_prior_
        assert prior is not None
        k = theta.shape[1]
        correct = (theta[:, np.arange(k), np.arange(k)] * prior).sum(axis=1)
        votes = (theta[:, :, :k].sum(axis=2) * prior).sum(axis=1)
        return np.divide(correct, votes, out=np.full_like(correct, np.nan), where=votes > 0)

    @property
    def confusion_(self) -> np.ndarray:
        """``(m, k, k+1)`` estimated ``P(lambda_j = o | Y = y)``; last column is abstain."""
        if self.log_theta_ is None:
            raise RuntimeError("model is not fitted")
        return np.exp(self.log_theta_)
