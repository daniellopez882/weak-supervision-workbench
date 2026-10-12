"""Selection strategies for the labelling queue.

Every strategy picks ``n`` rows from the unlabelled pool. Ties are broken uniformly at
random with the caller's generator, so a run is reproducible from its seed and no strategy
silently prefers low row ids.

* ``random``            - uniform over the unlabelled rows (the baseline).
* ``conflict_entropy``  - rows whose LF votes are most evenly split between classes.
* ``posterior_margin``  - rows where the label model's top two classes are closest.
* ``coreset``           - greedy k-center on TF-IDF vectors: the row farthest from
  everything labelled so far (Sener and Savarese, 2018).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import scipy.sparse as sp

from wsw.diagnostics import conflict_entropy


@dataclass(frozen=True)
class Pool:
    """What a strategy may look at: LF votes, optional features and the label model's view."""

    L: np.ndarray
    k: int
    features: sp.csr_matrix | None = None


class Strategy(Protocol):
    name: str

    def select(
        self,
        pool: Pool,
        posterior: np.ndarray,
        labelled: np.ndarray,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray: ...


def top_n(
    scores: np.ndarray, candidates: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Indices of the ``n`` highest-scoring candidates, exact ties in random order."""
    if n <= 0 or candidates.size == 0:
        return np.empty(0, dtype=np.int64)
    tiebreak = rng.random(candidates.size)
    order = np.lexsort((tiebreak, -scores[candidates]))
    return candidates[order[:n]].astype(np.int64)


def posterior_margin(P: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 posterior per row; 0 means the model cannot separate two classes."""
    part = np.sort(P, axis=1)
    return part[:, -1] - part[:, -2]


def _unlabelled(labelled: np.ndarray) -> np.ndarray:
    return np.flatnonzero(~labelled)


class RandomStrategy:
    name = "random"

    def select(
        self,
        pool: Pool,
        posterior: np.ndarray,
        labelled: np.ndarray,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        candidates = _unlabelled(labelled)
        n = min(n, candidates.size)
        return np.sort(rng.choice(candidates, size=n, replace=False)).astype(np.int64)


class ConflictEntropyStrategy:
    name = "conflict_entropy"

    def __init__(self) -> None:
        self._cache: tuple[np.ndarray, np.ndarray] | None = None

    def select(
        self,
        pool: Pool,
        posterior: np.ndarray,
        labelled: np.ndarray,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        # Votes only change when the LF set does (a new array), so scores are cached per array.
        if self._cache is None or self._cache[0] is not pool.L:
            self._cache = (pool.L, conflict_entropy(pool.L, pool.k))
        return top_n(self._cache[1], _unlabelled(labelled), n, rng)


class PosteriorMarginStrategy:
    name = "posterior_margin"

    def select(
        self,
        pool: Pool,
        posterior: np.ndarray,
        labelled: np.ndarray,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return top_n(-posterior_margin(posterior), _unlabelled(labelled), n, rng)


class CoresetStrategy:
    """Greedy k-center. Squared distances to the nearest labelled row are kept between calls
    and updated only for rows labelled since the last call, so each pick costs one sparse
    matrix-vector product instead of a full distance matrix."""

    name = "coreset"

    def __init__(self) -> None:
        self._min_dist: np.ndarray | None = None
        self._centres: set[int] = set()
        self._features: sp.csr_matrix | None = None

    def _add_centre(self, X: sp.csr_matrix, sq_norms: np.ndarray, i: int) -> None:
        assert self._min_dist is not None
        dots = np.asarray((X @ X[i].T).todense()).ravel()
        dist = np.maximum(sq_norms + sq_norms[i] - 2.0 * dots, 0.0)
        np.minimum(self._min_dist, dist, out=self._min_dist)
        self._centres.add(i)

    def select(
        self,
        pool: Pool,
        posterior: np.ndarray,
        labelled: np.ndarray,
        n: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        X = pool.features
        if X is None:
            raise ValueError("the coreset strategy needs feature vectors in the pool")
        sq_norms = np.asarray(X.multiply(X).sum(axis=1)).ravel()
        if self._features is not X:
            self._features = X
            self._min_dist = np.full(X.shape[0], np.inf)
            self._centres = set()
        for i in np.flatnonzero(labelled):
            if int(i) not in self._centres:
                self._add_centre(X, sq_norms, int(i))
        assert self._min_dist is not None
        taken = labelled.copy()
        picked: list[int] = []
        for _ in range(min(n, int((~taken).sum()))):
            candidates = np.flatnonzero(~taken)
            if not self._centres and not picked:
                choice = int(rng.choice(candidates))
            else:
                choice = int(top_n(self._min_dist, candidates, 1, rng)[0])
            picked.append(choice)
            taken[choice] = True
            self._add_centre(X, sq_norms, choice)
        return np.asarray(picked, dtype=np.int64)


STRATEGIES: dict[str, type[Strategy]] = {
    "random": RandomStrategy,
    "conflict_entropy": ConflictEntropyStrategy,
    "posterior_margin": PosteriorMarginStrategy,
    "coreset": CoresetStrategy,
}


def make_strategy(name: str) -> Strategy:
    try:
        return STRATEGIES[name]()
    except KeyError:
        raise ValueError(f"unknown strategy {name!r}; choose from {sorted(STRATEGIES)}") from None
