"""FlyingSquid (Fu et al., ICML 2020): closed-form LF accuracies from triplets.

For a binary task with votes ``s_i in {-1, 0, +1}`` and ``Y in {-1, +1}``, FlyingSquid's
model makes the accuracy variables ``s_i * Y`` mutually independent, so
``E[s_i s_j] = E[s_i Y] E[s_j Y] = a_i a_j``. Any three LFs then give
``|a_i| = sqrt(|E[s_i s_j] E[s_i s_l] / E[s_j s_l]|)``; we aggregate over every usable
pair ``(j, l)`` and take the positive root (LFs assumed better than random). Abstaining is
assumed independent of ``Y``, so ``P(s_i Y = +-1) = (c_i +- a_i) / 2`` with coverage
``c_i``. Multi-class tasks are split one-versus-rest and the per-class posteriors are
renormalised, the same reduction WRENCH applies to the reference implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Self

import numpy as np

from wsw.labelmodels.base import LabelModel, normalise_rows
from wsw.labelmodels.metal import BalanceSpec, class_balance_for
from wsw.lf import ABSTAIN, check_label_matrix

MAX_CONDITIONAL_ACCURACY = 0.999


@dataclass(frozen=True)
class BinaryTask:
    positive: int
    accuracy: np.ndarray  # a_i = E[s_i Y]
    coverage: np.ndarray
    prior_positive: float


def signed_votes(L: np.ndarray, positive: int) -> np.ndarray:
    return np.where(L == ABSTAIN, 0, np.where(L == positive, 1, -1)).astype(np.float64)


def triplet_accuracies(
    S: np.ndarray, *, aggregate: Literal["mean", "median"] = "mean", min_moment: float = 1e-6
) -> np.ndarray:
    """Estimate ``|E[s_i Y]|`` for every column of a signed vote matrix ``S``."""
    n, m = S.shape
    M = np.abs(S.T @ S / n)
    reduce = np.mean if aggregate == "mean" else np.median
    accuracy = np.zeros(m)
    for i in range(m):
        others = np.delete(np.arange(m), i)
        upper = np.triu_indices(others.size, 1)
        numer = np.outer(M[i, others], M[i, others])[upper]
        denom = M[np.ix_(others, others)][upper]
        usable = (denom > min_moment) & (numer > 0)
        if usable.any():
            accuracy[i] = float(reduce(np.sqrt(numer[usable] / denom[usable])))
        # No usable triplet: the LF never co-fires with a pair, so it stays uninformative (0).
    return accuracy


class FlyingSquid(LabelModel):
    name = "flyingsquid"

    def __init__(
        self,
        *,
        class_balance: BalanceSpec = "estimate",
        aggregate: Literal["mean", "median"] = "mean",
    ) -> None:
        super().__init__()
        self.class_balance = class_balance
        self.aggregate = aggregate
        self.tasks_: list[BinaryTask] = []
        self.class_balance_: np.ndarray | None = None
        self.class_balance_source_ = ""

    def fit(self, L: np.ndarray, k: int) -> Self:
        L = check_label_matrix(L, k)
        self.k = k
        p, self.class_balance_source_ = class_balance_for(self.class_balance, L, k)
        self.class_balance_ = p
        positives = [1] if k == 2 else list(range(k))
        self.tasks_ = []
        for positive in positives:
            S = signed_votes(L, positive)
            coverage = (S != 0).mean(axis=0)
            accuracy = np.minimum(
                triplet_accuracies(S, aggregate=self.aggregate),
                coverage * (2 * MAX_CONDITIONAL_ACCURACY - 1),
            )
            self.tasks_.append(BinaryTask(positive, accuracy, coverage, float(p[positive])))
        return self

    def lf_accuracy(self) -> np.ndarray | None:
        """Binary tasks only: ``a = c (2 acc - 1)`` inverted per LF. The one-vs-rest tasks of
        a multi-class fit estimate a different quantity, so they report ``None``."""
        if self.k != 2 or not self.tasks_:
            return None
        task = self.tasks_[0]
        c, a = task.coverage, task.accuracy
        return np.divide(a, c, out=np.full_like(a, np.nan), where=c > 0) * 0.5 + 0.5

    @staticmethod
    def _positive_proba(task: BinaryTask, S: np.ndarray) -> np.ndarray:
        c, a = task.coverage, task.accuracy
        with np.errstate(divide="ignore", invalid="ignore"):
            weight = np.where(c > 0, np.log((c + a) / (c - a)), 0.0)
        log_odds = np.log(task.prior_positive / (1 - task.prior_positive)) + S @ weight
        return 1.0 / (1.0 + np.exp(-log_odds))

    def _proba(self, L: np.ndarray) -> np.ndarray:
        assert self.k is not None
        if self.tasks_ and L.shape[1] != self.tasks_[0].accuracy.shape[0]:
            raise ValueError("label matrix has a different number of LFs than at fit time")
        if self.k == 2:
            pos = self._positive_proba(self.tasks_[0], signed_votes(L, 1))
            return np.column_stack([1.0 - pos, pos])
        scores = np.column_stack(
            [self._positive_proba(t, signed_votes(L, t.positive)) for t in self.tasks_]
        )
        return normalise_rows(scores)
