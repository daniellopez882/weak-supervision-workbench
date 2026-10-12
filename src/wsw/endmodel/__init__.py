"""Noise-aware end model: TF-IDF features + logistic regression on soft labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from wsw.data.dataset import Dataset
from wsw.endmodel.logreg import SoftLabelLogisticRegression
from wsw.endmodel.tfidf import TfidfVectorizer
from wsw.lf import ABSTAIN


@dataclass(frozen=True)
class TextFeatures:
    """TF-IDF matrices for all three splits, fitted on the training texts only."""

    vectorizer: TfidfVectorizer
    train: sp.csr_matrix
    valid: sp.csr_matrix
    test: sp.csr_matrix

    @classmethod
    def build(cls, dataset: Dataset, **vectorizer_kwargs: object) -> TextFeatures:
        vec = TfidfVectorizer(**vectorizer_kwargs)  # type: ignore[arg-type]
        train = vec.fit_transform(dataset.train.texts)
        return cls(
            vec, train, vec.transform(dataset.valid.texts), vec.transform(dataset.test.texts)
        )


def training_targets(
    L: np.ndarray,
    posteriors: np.ndarray,
    gold_idx: np.ndarray | None = None,
    gold_y: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rows the end model trains on and their soft targets.

    Rows where every LF abstains carry no weak signal and are dropped (as in WRENCH),
    unless a human labelled them; human labels replace the posterior with a one-hot row.
    """
    covered = (L != ABSTAIN).any(axis=1)
    targets = posteriors.copy()
    if gold_idx is not None and gold_y is not None and len(gold_idx):
        covered[gold_idx] = True
        targets[gold_idx] = np.eye(posteriors.shape[1])[gold_y]
    rows = np.flatnonzero(covered)
    return rows, targets[rows]


def train_end_model(
    X: sp.csr_matrix,
    rows: np.ndarray,
    targets: np.ndarray,
    *,
    C: float = 1.0,
    warm_start: SoftLabelLogisticRegression | None = None,
) -> SoftLabelLogisticRegression:
    return SoftLabelLogisticRegression(C=C).fit(X[rows], targets, warm_start=warm_start)


__all__ = [
    "SoftLabelLogisticRegression",
    "TextFeatures",
    "TfidfVectorizer",
    "train_end_model",
    "training_targets",
]
