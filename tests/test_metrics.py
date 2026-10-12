from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from sklearn.metrics import accuracy_score, f1_score

from wsw.metrics import accuracy, f1, normalised_aulc


@given(st.integers(2, 5), st.integers(1, 60), st.integers(0, 10_000))
def test_metrics_match_scikit_learn(k: int, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    y, p = rng.integers(0, k, n), rng.integers(0, k, n)
    assert accuracy(y, p) == pytest.approx(accuracy_score(y, p))
    average = "binary" if k == 2 else "macro"
    expected = f1_score(y, p, average=average, labels=list(range(k)), zero_division=0)
    assert f1(y, p, k) == pytest.approx(expected)


def test_normalised_aulc() -> None:
    flat = normalised_aulc(np.array([25, 50, 500]), np.array([0.7, 0.7, 0.7]))
    assert flat == pytest.approx(0.7)
    # A straight rise from 0.5 to 1.0 averages 0.75.
    assert normalised_aulc(np.array([0, 100]), np.array([0.5, 1.0])) == pytest.approx(0.75)
    with pytest.raises(ValueError):
        normalised_aulc(np.array([10, 5]), np.array([0.1, 0.2]))
