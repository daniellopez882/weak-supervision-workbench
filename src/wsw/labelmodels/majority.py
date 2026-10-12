"""Majority vote: the unweighted baseline every label model has to beat."""

from __future__ import annotations

from typing import Self

import numpy as np

from wsw.labelmodels.base import LabelModel, normalise_rows
from wsw.lf import check_label_matrix, vote_counts


class MajorityVote(LabelModel):
    name = "majority_vote"

    def fit(self, L: np.ndarray, k: int) -> Self:
        check_label_matrix(L, k)
        self.k = k
        return self

    def _proba(self, L: np.ndarray) -> np.ndarray:
        assert self.k is not None
        # Vote shares (as in WRENCH); argmax with random ties gives the hard majority vote,
        # and rows without votes become uniform.
        return normalise_rows(vote_counts(L, self.k).astype(np.float64))
