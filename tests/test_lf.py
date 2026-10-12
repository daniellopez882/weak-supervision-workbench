from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from wsw.lf import ABSTAIN, KeywordLF, PythonLF, RegexLF, apply_lfs, one_hot, vote_counts


def test_keyword_lf_matches_whole_words_and_phrases() -> None:
    lf = KeywordLF("kw", ("new york", "app"), 1)
    assert lf("I moved to New York.") == 1
    assert lf("the App crashed") == 1
    assert lf("happy days") == ABSTAIN  # "app" inside a word does not count
    assert lf("newyork") == ABSTAIN


def test_keyword_lf_rejects_empty_keywords() -> None:
    with pytest.raises(ValueError, match="at least one keyword"):
        KeywordLF("kw", (" ", ""), 0)


def test_regex_lf_and_invalid_pattern() -> None:
    lf = RegexLF("money", r"\$\d+", 1)
    assert lf("win $100 now") == 1
    assert lf("win money") == ABSTAIN
    with pytest.raises(ValueError, match="invalid regular expression"):
        RegexLF("bad", "(", 0)


def test_apply_lfs_builds_matrix_and_checks_range() -> None:
    lfs = [KeywordLF("a", ("spam",), 1), PythonLF("short", lambda t: 0 if len(t) < 5 else -1)]
    L = apply_lfs(lfs, ["spam here", "hi", "nothing"], k=2)
    np.testing.assert_array_equal(L, [[1, ABSTAIN], [ABSTAIN, 0], [ABSTAIN, ABSTAIN]])
    with pytest.raises(ValueError, match="returned 5"):
        apply_lfs([PythonLF("bad", lambda t: 5)], ["x"], k=2)


@given(
    arrays(np.int64, st.tuples(st.integers(1, 30), st.integers(1, 6)), elements=st.integers(-1, 3))
)
def test_vote_counts_and_one_hot_agree(L: np.ndarray) -> None:
    k = 4
    counts = vote_counts(L, k)
    assert counts.sum() == (L != ABSTAIN).sum()
    np.testing.assert_array_equal(one_hot(L, k, abstain_column=False).sum(axis=1), counts)
    with_abstain = one_hot(L, k, abstain_column=True)
    np.testing.assert_array_equal(with_abstain.sum(axis=2), np.ones(L.shape))
    np.testing.assert_array_equal(with_abstain[:, :, k], L == ABSTAIN)
