from __future__ import annotations

import random

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from wsw.data.synthetic import SyntheticLabels
from wsw.labelmodels import (
    DawidSkene,
    FlyingSquid,
    LabelModel,
    MajorityVote,
    MeTaL,
    argmax_random_ties,
    balance_from_labels,
    estimate_class_balance,
    make_label_model,
)
from wsw.labelmodels.flyingsquid import triplet_accuracies
from wsw.labelmodels.metal import MetalObjective
from wsw.lf import ABSTAIN


def _accuracy_on(data: SyntheticLabels, model: LabelModel) -> float:
    pred = model.predict(data.L, np.random.default_rng(0))
    covered = (data.L != ABSTAIN).any(axis=1)
    return float((pred[covered] == data.y[covered]).mean())


def test_dawid_skene_recovers_lf_accuracies_and_prior(binary_synthetic: SyntheticLabels) -> None:
    model = DawidSkene().fit(binary_synthetic.L, 2)
    np.testing.assert_allclose(model.lf_accuracy(), binary_synthetic.accuracy, atol=0.03)
    np.testing.assert_allclose(model.class_prior_, binary_synthetic.balance, atol=0.03)
    # With abstain as an outcome, coverage is part of each confusion matrix.
    np.testing.assert_allclose(
        1 - model.confusion_[:, :, 2].mean(axis=1), binary_synthetic.coverage, atol=0.02
    )


def test_dawid_skene_multiclass(multiclass_synthetic: SyntheticLabels) -> None:
    model = DawidSkene().fit(multiclass_synthetic.L, 3)
    np.testing.assert_allclose(model.lf_accuracy(), multiclass_synthetic.accuracy, atol=0.03)
    ll = np.asarray(model.log_likelihood_)
    assert np.all(np.diff(ll) >= -1e-6 * np.abs(ll[1:]))  # EM does not decrease it


def test_dawid_skene_clamps_gold_rows(binary_synthetic: SyntheticLabels) -> None:
    L = binary_synthetic.L[:2000]
    idx = np.array([0, 5, 9])
    gold = 1 - binary_synthetic.y[idx]
    model = DawidSkene().fit(L, 2, gold_idx=idx, gold_y=gold)
    unclamped = DawidSkene().fit(L, 2)
    # Flipped gold labels pull the estimated confusion matrices away from the unclamped fit.
    assert not np.allclose(model.confusion_, unclamped.confusion_)
    with pytest.raises(ValueError, match="same length"):
        DawidSkene().fit(L, 2, gold_idx=idx, gold_y=gold[:2])


def test_dawid_skene_starts_uncovered_rows_from_covered_class_mix() -> None:
    # Many rare spam LFs, one ham LF and 80% of rows without votes (an SMS-like matrix). A
    # uniform start lets EM explain "nobody fired" as a class; rows without votes must stay
    # with the majority class.
    rng = np.random.default_rng(3)
    n, m = 5000, 30
    y = (rng.random(n) < 0.12).astype(int)
    fires = ((rng.random((n, m)) < 0.08) & (y[:, None] == 1)) | (rng.random((n, m)) < 0.004)
    L = np.where(fires, 1, ABSTAIN)
    L[:, 0] = np.where((rng.random(n) < 0.3) & (y == 0), 0, ABSTAIN)
    model = DawidSkene().fit(L, 2)
    nobody = (L == ABSTAIN).all(axis=1)
    assert nobody.mean() > 0.5
    assert model.predict_proba(L[nobody][:1])[0, 0] > 0.5


def test_metal_recovers_accuracies_with_known_balance(binary_synthetic: SyntheticLabels) -> None:
    model = MeTaL(class_balance=binary_synthetic.balance, l2=0.0).fit(binary_synthetic.L, 2)
    np.testing.assert_allclose(model.lf_accuracy(), binary_synthetic.accuracy, atol=0.04)
    assert _accuracy_on(binary_synthetic, model) > 0.85


def test_metal_multiclass(multiclass_synthetic: SyntheticLabels) -> None:
    model = MeTaL(class_balance=multiclass_synthetic.balance, l2=0.0)
    model.fit(multiclass_synthetic.L, 3)
    np.testing.assert_allclose(model.lf_accuracy(), multiclass_synthetic.accuracy, atol=0.05)


def test_metal_gradient_matches_finite_differences(binary_synthetic: SyntheticLabels) -> None:
    obj = MetalObjective(binary_synthetic.L[:3000], 2, binary_synthetic.balance, l2=0.1)
    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 0.9, size=obj.mu_init.size)
    _, grad = obj.value_and_grad(x)
    eps = 1e-6
    for i in rng.choice(x.size, size=8, replace=False):
        step = np.zeros_like(x)
        step[i] = eps
        numeric = (obj.value_and_grad(x + step)[0] - obj.value_and_grad(x - step)[0]) / (2 * eps)
        assert grad[i] == pytest.approx(numeric, rel=1e-4, abs=1e-8)


def test_flyingsquid_triplets_are_exact_on_population_moments() -> None:
    # Every vote pattern of three always-voting independent LFs, repeated in proportion to
    # its exact probability under a balanced prior.
    acc = np.array([0.9, 0.7, 0.8])
    rows, counts = [], []
    for y in (-1, 1):
        for pattern in np.ndindex(2, 2, 2):
            rows.append([y if right else -y for right in pattern])
            p = 0.5 * np.prod(
                [a if right else 1 - a for a, right in zip(acc, pattern, strict=True)]
            )
            counts.append(round(p * 100_000))
    S = np.repeat(np.asarray(rows, dtype=float), counts, axis=0)
    np.testing.assert_allclose(triplet_accuracies(S), 2 * acc - 1, atol=1e-3)


def test_flyingsquid_recovers_binary_accuracies(binary_synthetic: SyntheticLabels) -> None:
    model = FlyingSquid(class_balance=binary_synthetic.balance).fit(binary_synthetic.L, 2)
    np.testing.assert_allclose(model.lf_accuracy(), binary_synthetic.accuracy, atol=0.05)
    assert _accuracy_on(binary_synthetic, model) > 0.85


def test_flyingsquid_multiclass_predicts_without_accuracy(
    multiclass_synthetic: SyntheticLabels,
) -> None:
    model = FlyingSquid(class_balance=multiclass_synthetic.balance)
    model.fit(multiclass_synthetic.L, 3)
    assert model.lf_accuracy() is None
    assert _accuracy_on(multiclass_synthetic, model) > 0.8


def test_class_balance_estimate_recovers_prior(
    binary_synthetic: SyntheticLabels, multiclass_synthetic: SyntheticLabels
) -> None:
    est = estimate_class_balance(binary_synthetic.L, 2)
    np.testing.assert_allclose(est.balance, binary_synthetic.balance, atol=0.05)
    est3 = estimate_class_balance(multiclass_synthetic.L, 3)
    np.testing.assert_allclose(est3.balance, multiclass_synthetic.balance, atol=0.06)
    with pytest.raises(ValueError, match="three"):
        estimate_class_balance(binary_synthetic.L[:, :2], 2)


def test_balance_from_labels_floors_missing_classes() -> None:
    p = balance_from_labels(np.array([0, 0, 1, 1]), 3)
    assert p[2] > 0
    assert p.sum() == pytest.approx(1.0)
    assert p[0] == pytest.approx(p[1])


def test_majority_vote_shares_and_random_ties() -> None:
    L = np.array([[0, 0, 1], [1, ABSTAIN, ABSTAIN], [0, 1, ABSTAIN], [ABSTAIN] * 3])
    P = MajorityVote().fit(L, 2).predict_proba(L)
    np.testing.assert_allclose(P, [[2 / 3, 1 / 3], [0, 1], [0.5, 0.5], [0.5, 0.5]])
    picks = {int(argmax_random_ties(P[2:3], np.random.default_rng(s))[0]) for s in range(20)}
    assert picks == {0, 1}


def test_argmax_random_ties_is_seeded() -> None:
    P = np.full((50, 3), 1 / 3)
    a = argmax_random_ties(P, np.random.default_rng(4))
    b = argmax_random_ties(P, np.random.default_rng(4))
    np.testing.assert_array_equal(a, b)


def test_unknown_label_model_name() -> None:
    with pytest.raises(ValueError, match="unknown label model"):
        make_label_model("nope")


small_matrices = arrays(
    np.int64, st.tuples(st.integers(8, 40), st.integers(3, 6)), elements=st.integers(-1, 2)
)


@given(small_matrices, st.sampled_from(["majority_vote", "dawid_skene", "metal", "flyingsquid"]))
def test_posteriors_are_distributions(L: np.ndarray, name: str) -> None:
    model = make_label_model(name, "uniform").fit(L, 3)
    P = model.predict_proba(L)
    assert P.shape == (L.shape[0], 3)
    assert np.all(P >= -1e-12)
    assert np.all(P <= 1 + 1e-12)
    np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=1e-9)


@given(small_matrices, st.randoms(use_true_random=False))
def test_dawid_skene_does_not_depend_on_lf_order(L: np.ndarray, rnd: random.Random) -> None:
    perm = list(range(L.shape[1]))
    rnd.shuffle(perm)
    a = DawidSkene().fit(L, 3).predict_proba(L)
    b = DawidSkene().fit(L[:, perm], 3).predict_proba(L[:, perm])
    np.testing.assert_allclose(a, b, atol=1e-6)


@pytest.mark.reference
def test_metal_fit_is_at_least_as_good_as_snorkel_on_its_own_objective(
    binary_synthetic: SyntheticLabels,
) -> None:
    snorkel_model = pytest.importorskip("snorkel.labeling.model")
    L, p = binary_synthetic.L.astype(np.int64), binary_synthetic.balance
    ref = snorkel_model.LabelModel(cardinality=2, verbose=False)
    ref.fit(L, class_balance=p, n_epochs=200, seed=0, progress_bar=False)
    ours = MeTaL(class_balance=p, l2=0.0).fit(L, 2)
    objective = MetalObjective(L, 2, p)
    snorkel_mu = ref.get_conditional_probs()[:, 1:, :].reshape(-1, 2)
    assert ours.loss_ <= objective.value(snorkel_mu) + 1e-9
    agreement = (ours.predict_proba(L).argmax(1) == ref.predict_proba(L).argmax(1)).mean()
    assert agreement > 0.97
