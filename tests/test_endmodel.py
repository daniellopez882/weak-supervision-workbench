from __future__ import annotations

import numpy as np
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidf
from sklearn.linear_model import LogisticRegression

from wsw.data import load_dataset
from wsw.endmodel import (
    SoftLabelLogisticRegression,
    TextFeatures,
    TfidfVectorizer,
    train_end_model,
    training_targets,
)
from wsw.lf import ABSTAIN

TEXTS = [
    "The app crashes on login every time",
    "Please add a dark mode option",
    "Love the new update, great work",
    "Login fails and the app crashes again",
    "Would love an export option for reports",
    "Great app, the best calendar view",
    "sync is broken and slow",
]


def test_tfidf_matches_scikit_learn() -> None:
    ours = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True).fit_transform(TEXTS)
    ref = SkTfidf(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    theirs = ref.fit_transform(TEXTS)
    # Same vocabulary in the same (sorted) order, so the matrices compare entry by entry.
    assert list(ref.get_feature_names_out()) == sorted(ref.vocabulary_)
    np.testing.assert_allclose(ours.toarray(), theirs.toarray(), atol=1e-12)


def test_tfidf_max_features_and_unknown_terms() -> None:
    vec = TfidfVectorizer(ngram_range=(1, 1), min_df=1, max_features=3).fit(TEXTS)
    assert len(vec.vocabulary_) == 3
    X = vec.transform(["completely unrelated words"])
    assert X.nnz == 0
    with pytest.raises(ValueError, match="empty vocabulary"):
        TfidfVectorizer(min_df=5).fit(["one text", "two texts"])


def test_soft_label_lr_matches_scikit_learn_on_hard_labels() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.normal(scale=0.5, size=300) > 0).astype(int)
    y[y == 1] += (X[y == 1, 2] > 0.8).astype(int)  # three classes
    ours = SoftLabelLogisticRegression(C=0.5, tol=1e-8).fit(X, np.eye(3)[y])
    ref = LogisticRegression(C=0.5, tol=1e-10, max_iter=5000).fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-4)


def test_soft_label_lr_rejects_bad_targets() -> None:
    X = np.ones((3, 2))
    with pytest.raises(ValueError, match="sum to one"):
        SoftLabelLogisticRegression().fit(X, np.full((3, 2), 0.3))
    with pytest.raises(RuntimeError):
        SoftLabelLogisticRegression().predict(X)


def test_soft_labels_shift_the_decision() -> None:
    X = np.eye(2)
    confident = SoftLabelLogisticRegression(C=10).fit(X, np.array([[0.95, 0.05], [0.05, 0.95]]))
    unsure = SoftLabelLogisticRegression(C=10).fit(X, np.array([[0.6, 0.4], [0.4, 0.6]]))
    assert confident.predict_proba(X)[0, 0] > unsure.predict_proba(X)[0, 0] > 0.5


def test_training_targets_drop_uncovered_rows_unless_labelled() -> None:
    L = np.array([[0, ABSTAIN], [ABSTAIN, ABSTAIN], [1, 1], [ABSTAIN, ABSTAIN]])
    P = np.full((4, 2), 0.5)
    rows, targets = training_targets(L, P)
    np.testing.assert_array_equal(rows, [0, 2])
    rows, targets = training_targets(L, P, np.array([3, 2]), np.array([1, 0]))
    np.testing.assert_array_equal(rows, [0, 2, 3])
    np.testing.assert_array_equal(targets[1:], [[1, 0], [0, 1]])


def test_end_model_learns_demo_task() -> None:
    ds = load_dataset("demo")
    feats = TextFeatures.build(ds, min_df=1)
    rows = np.arange(len(ds.train))
    model = train_end_model(feats.train, rows, np.eye(ds.k)[ds.train.y])
    # About an eighth of demo reviews carry no class cue at all, which caps accuracy near 0.9.
    assert (model.predict(feats.test) == ds.test.y).mean() > 0.8
