"""Labelling functions and label-matrix construction.

A labelling function (LF) maps one text to a class id in ``[0, k)`` or to ``ABSTAIN``.
Applying ``m`` LFs to ``n`` texts gives the ``(n, m)`` label matrix every label model consumes.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

ABSTAIN = -1
LABEL_DTYPE = np.int16


@runtime_checkable
class LabelingFunction(Protocol):
    name: str

    def __call__(self, text: str) -> int: ...


@dataclass(frozen=True)
class KeywordLF:
    """Votes ``label`` when any keyword occurs as a whole word or phrase (case-insensitive)."""

    name: str
    keywords: tuple[str, ...]
    label: int
    _pattern: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        cleaned = tuple(k.strip().lower() for k in self.keywords if k.strip())
        if not cleaned:
            raise ValueError(f"LF {self.name!r} needs at least one keyword")
        object.__setattr__(self, "keywords", cleaned)
        # Longest first so that "new york" wins over "new" inside the alternation.
        alternation = "|".join(re.escape(k) for k in sorted(cleaned, key=len, reverse=True))
        object.__setattr__(self, "_pattern", re.compile(rf"(?<!\w)(?:{alternation})(?!\w)"))

    def __call__(self, text: str) -> int:
        return self.label if self._pattern.search(text.lower()) else ABSTAIN


@dataclass(frozen=True)
class RegexLF:
    """Votes ``label`` when the regular expression matches anywhere in the text."""

    name: str
    pattern: str
    label: int
    ignore_case: bool = True
    _compiled: re.Pattern[str] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        flags = re.IGNORECASE if self.ignore_case else 0
        try:
            compiled = re.compile(self.pattern, flags)
        except re.error as exc:
            raise ValueError(f"LF {self.name!r}: invalid regular expression: {exc}") from exc
        object.__setattr__(self, "_compiled", compiled)

    def __call__(self, text: str) -> int:
        return self.label if self._compiled.search(text) else ABSTAIN


@dataclass(frozen=True)
class PythonLF:
    """Wraps an arbitrary callable written in code (never built from UI input)."""

    name: str
    fn: Callable[[str], int]

    def __call__(self, text: str) -> int:
        return int(self.fn(text))


def apply_lfs(lfs: Sequence[LabelingFunction], texts: Sequence[str], k: int) -> np.ndarray:
    """Build the ``(n, m)`` label matrix; raises if an LF returns an out-of-range vote."""
    L = np.full((len(texts), len(lfs)), ABSTAIN, dtype=LABEL_DTYPE)
    for j, lf in enumerate(lfs):
        column = np.fromiter((lf(t) for t in texts), dtype=np.int64, count=len(texts))
        bad = (column < ABSTAIN) | (column >= k)
        if bad.any():
            first = int(np.flatnonzero(bad)[0])
            raise ValueError(f"LF {lf.name!r} returned {column[first]} for row {first}; k={k}")
        L[:, j] = column
    return L


def check_label_matrix(L: np.ndarray, k: int) -> np.ndarray:
    """Validate shape and value range and return the matrix as int64."""
    L = np.asarray(L)
    if L.ndim != 2:
        raise ValueError(f"label matrix must be 2-D, got shape {L.shape}")
    if L.size and (L.min() < ABSTAIN or L.max() >= k):
        raise ValueError(f"label matrix values must lie in [-1, {k - 1}]")
    return L.astype(np.int64, copy=False)


def vote_counts(L: np.ndarray, k: int) -> np.ndarray:
    """``(n, k)`` number of LFs voting for each class per row (abstains ignored)."""
    L = check_label_matrix(L, k)
    counts = np.zeros((L.shape[0], k), dtype=np.int64)
    rows, cols = np.nonzero(L != ABSTAIN)
    np.add.at(counts, (rows, L[rows, cols]), 1)
    return counts


def one_hot(L: np.ndarray, k: int, *, abstain_column: bool) -> np.ndarray:
    """``(n, m, k)`` indicators, or ``(n, m, k + 1)`` with abstain as the last outcome."""
    L = check_label_matrix(L, k)
    width = k + 1 if abstain_column else k
    out = np.zeros((*L.shape, width), dtype=np.float64)
    idx = np.where(L == ABSTAIN, k, L) if abstain_column else L
    rows, cols = np.nonzero(idx != ABSTAIN) if not abstain_column else np.indices(L.shape)
    rows, cols = rows.ravel(), cols.ravel()
    out[rows, cols, idx[rows, cols]] = 1.0
    return out
