"""LF diagnostics: coverage, overlap and conflict per LF and per pair, plus row-level conflict.

Per-LF definitions follow Snorkel's ``LFAnalysis`` so the numbers can be checked against it:

* coverage  - fraction of rows on which the LF votes;
* overlap   - fraction of rows on which it votes and at least one other LF votes;
* conflict  - fraction of rows on which it votes and another LF votes a different class.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from wsw.lf import ABSTAIN, check_label_matrix, vote_counts


@dataclass(frozen=True)
class LFStats:
    name: str
    polarity: tuple[int, ...]
    coverage: float
    overlap: float
    conflict: float
    #: Accuracy on rows with a known label where the LF voted; ``None`` without such rows.
    empirical_accuracy: float | None
    n_labelled_votes: int


@dataclass(frozen=True)
class PairMatrices:
    """``(m, m)`` fractions of rows; the diagonal of ``overlap`` is each LF's coverage."""

    overlap: np.ndarray
    conflict: np.ndarray


def lf_stats(
    L: np.ndarray,
    k: int,
    names: Sequence[str],
    gold_idx: np.ndarray | None = None,
    gold_y: np.ndarray | None = None,
) -> list[LFStats]:
    L = check_label_matrix(L, k)
    n, m = L.shape
    if len(names) != m:
        raise ValueError(f"{m} LF columns but {len(names)} names")
    votes = L != ABSTAIN
    counts = vote_counts(L, k)
    n_votes = votes.sum(axis=1)
    gold_idx = np.asarray([] if gold_idx is None else gold_idx, dtype=np.int64)
    gold_y = np.asarray([] if gold_y is None else gold_y, dtype=np.int64)
    denom = max(n, 1)
    out: list[LFStats] = []
    for j in range(m):
        col = L[:, j]
        fired = votes[:, j]
        others = n_votes - fired
        # Another LF disagrees iff not every other vote on the row is for this LF's class.
        same = counts[np.arange(n), np.where(fired, col, 0)] - 1
        conflicted = fired & (others > same)
        gold_votes = col[gold_idx] != ABSTAIN
        n_gold = int(gold_votes.sum())
        out.append(
            LFStats(
                name=names[j],
                polarity=tuple(int(c) for c in np.unique(col[fired])),
                coverage=float(fired.sum() / denom),
                overlap=float((fired & (others > 0)).sum() / denom),
                conflict=float(conflicted.sum() / denom),
                empirical_accuracy=(
                    float((col[gold_idx][gold_votes] == gold_y[gold_votes]).mean())
                    if n_gold
                    else None
                ),
                n_labelled_votes=n_gold,
            )
        )
    return out


def pair_matrices(L: np.ndarray, k: int) -> PairMatrices:
    L = check_label_matrix(L, k)
    n = max(L.shape[0], 1)
    fired = (L != ABSTAIN).astype(np.float64)
    overlap = fired.T @ fired / n
    # Agreements summed over classes; overlap minus agreement is disagreement.
    agree = np.zeros_like(overlap)
    for c in range(k):
        is_c = (c == L).astype(np.float64)
        agree += is_c.T @ is_c
    return PairMatrices(overlap=overlap, conflict=overlap - agree / n)


def conflict_entropy(L: np.ndarray, k: int) -> np.ndarray:
    """Entropy of each row's vote shares divided by ``log k``; 0 for unanimous or empty rows."""
    counts = vote_counts(L, k).astype(np.float64)
    total = counts.sum(axis=1, keepdims=True)
    shares = np.divide(counts, total, out=np.zeros_like(counts), where=total > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(shares > 0, shares * np.log(shares), 0.0)
    return -terms.sum(axis=1) / np.log(k)


@dataclass(frozen=True)
class MatrixSummary:
    n_rows: int
    n_lfs: int
    coverage: float
    mean_votes_per_covered_row: float
    conflict_rate: float


def matrix_summary(L: np.ndarray, k: int) -> MatrixSummary:
    L = check_label_matrix(L, k)
    n_votes = (L != ABSTAIN).sum(axis=1)
    covered = n_votes > 0
    return MatrixSummary(
        n_rows=L.shape[0],
        n_lfs=L.shape[1],
        coverage=float(covered.mean()) if L.shape[0] else 0.0,
        mean_votes_per_covered_row=float(n_votes[covered].mean()) if covered.any() else 0.0,
        conflict_rate=float((conflict_entropy(L, k) > 0).mean()) if L.shape[0] else 0.0,
    )
