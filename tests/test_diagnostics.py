from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from wsw.diagnostics import conflict_entropy, lf_stats, matrix_summary, pair_matrices
from wsw.lf import ABSTAIN

A = ABSTAIN
# Hand-checked fixture: 5 rows, 3 LFs, k = 3.
L = np.array(
    [
        [0, 0, A],  # agreement
        [0, 1, A],  # conflict between LF0 and LF1
        [A, 2, 2],  # agreement
        [1, A, 2],  # conflict between LF0 and LF2
        [A, A, A],  # uncovered
    ]
)
NAMES = ["lf0", "lf1", "lf2"]


def test_lf_stats_on_fixture() -> None:
    stats = lf_stats(L, 3, NAMES, gold_idx=np.array([0, 1, 3]), gold_y=np.array([0, 0, 2]))
    s0, s1, s2 = stats
    assert (s0.coverage, s0.overlap, s0.conflict) == (0.6, 0.6, 0.4)
    assert (s1.coverage, s1.overlap, s1.conflict) == (0.6, 0.6, 0.2)
    assert (s2.coverage, s2.overlap, s2.conflict) == (0.4, 0.4, 0.2)
    assert s0.polarity == (0, 1)
    assert s1.polarity == (0, 1, 2)
    # LF0 voted on gold rows 0, 1, 3 and was right on 0 and 1.
    assert s0.empirical_accuracy == pytest.approx(2 / 3)
    assert s0.n_labelled_votes == 3
    # LF2 voted on gold row 3 only, correctly.
    assert s2.empirical_accuracy == 1.0
    assert lf_stats(L, 3, NAMES)[0].empirical_accuracy is None


def test_pair_matrices_on_fixture() -> None:
    pm = pair_matrices(L, 3)
    np.testing.assert_allclose(np.diag(pm.overlap), [0.6, 0.6, 0.4])
    assert pm.overlap[0, 1] == pytest.approx(0.4)
    assert pm.conflict[0, 1] == pytest.approx(0.2)
    assert pm.conflict[1, 2] == pytest.approx(0.0)
    assert pm.conflict[0, 2] == pytest.approx(0.2)
    np.testing.assert_allclose(np.diag(pm.conflict), 0.0)


def test_conflict_entropy_hand_values() -> None:
    rows = np.array(
        [
            [0, 1, A],  # split 1:1 between two of three classes
            [0, 1, 2],  # uniform over all three classes
            [0, 0, 1],  # 2:1
            [2, 2, 2],  # unanimous
            [A, A, A],  # no votes
        ]
    )
    h = conflict_entropy(rows, 3)
    expected_21 = -(2 / 3 * math.log(2 / 3) + 1 / 3 * math.log(1 / 3)) / math.log(3)
    np.testing.assert_allclose(h, [math.log(2) / math.log(3), 1.0, expected_21, 0.0, 0.0])
    assert conflict_entropy(np.array([[0, 1]]), 2)[0] == pytest.approx(1.0)


def test_matrix_summary() -> None:
    s = matrix_summary(L, 3)
    assert (s.n_rows, s.n_lfs) == (5, 3)
    assert s.coverage == pytest.approx(0.8)
    assert s.mean_votes_per_covered_row == pytest.approx(2.0)
    assert s.conflict_rate == pytest.approx(0.4)


@given(
    arrays(np.int64, st.tuples(st.integers(1, 25), st.integers(1, 5)), elements=st.integers(-1, 2))
)
def test_diagnostic_invariants(M: np.ndarray) -> None:
    h = conflict_entropy(M, 3)
    assert np.all((h >= -1e-12) & (h <= 1 + 1e-12))
    for s in lf_stats(M, 3, [f"lf{j}" for j in range(M.shape[1])]):
        assert 0 <= s.conflict <= s.overlap <= s.coverage <= 1
    pm = pair_matrices(M, 3)
    np.testing.assert_allclose(pm.overlap, pm.overlap.T)
    assert np.all(pm.conflict <= pm.overlap + 1e-12)


@pytest.mark.reference
def test_lf_stats_match_snorkel_lf_analysis() -> None:
    lf_analysis = pytest.importorskip("snorkel.labeling").LFAnalysis
    rng = np.random.default_rng(1)
    M = rng.integers(-1, 3, size=(200, 5))
    ours = lf_stats(M, 3, [f"lf{j}" for j in range(5)])
    ref = lf_analysis(M.astype(np.int64))
    np.testing.assert_allclose([s.coverage for s in ours], ref.lf_coverages())
    np.testing.assert_allclose([s.overlap for s in ours], ref.lf_overlaps())
    np.testing.assert_allclose([s.conflict for s in ours], ref.lf_conflicts())
