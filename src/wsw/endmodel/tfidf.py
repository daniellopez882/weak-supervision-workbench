"""TF-IDF features written out explicitly (no scikit-learn in the training path).

Settings follow scikit-learn's ``TfidfVectorizer`` conventions so the test-suite can
check parity: lower-casing, the ``\\b\\w\\w+\\b`` token pattern, smoothed idf
``ln((1 + n) / (1 + df)) + 1``, optional sublinear tf and L2-normalised rows.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Self

import numpy as np
import scipy.sparse as sp

TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def ngrams(text: str, ngram_range: tuple[int, int]) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    low, high = ngram_range
    out: list[str] = []
    for size in range(low, high + 1):
        if size == 1:
            out.extend(tokens)
        else:
            out.extend(" ".join(tokens[i : i + size]) for i in range(len(tokens) - size + 1))
    return out


class TfidfVectorizer:
    def __init__(
        self,
        *,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_features: int | None = None,
        sublinear_tf: bool = True,
    ) -> None:
        if ngram_range[0] < 1 or ngram_range[1] < ngram_range[0]:
            raise ValueError(f"invalid ngram_range {ngram_range}")
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.vocabulary_: dict[str, int] = {}
        self.idf_: np.ndarray = np.empty(0)

    def fit(self, texts: Sequence[str]) -> Self:
        df: Counter[str] = Counter()
        for text in texts:
            df.update(set(ngrams(text, self.ngram_range)))
        kept = [(term, count) for term, count in df.items() if count >= self.min_df]
        if self.max_features is not None and len(kept) > self.max_features:
            # Most frequent by document frequency, ties broken alphabetically for determinism.
            kept = sorted(kept, key=lambda tc: (-tc[1], tc[0]))[: self.max_features]
        terms = sorted(term for term, _ in kept)
        if not terms:
            raise ValueError("empty vocabulary; lower min_df or provide more text")
        self.vocabulary_ = {term: i for i, term in enumerate(terms)}
        doc_freq = np.array([df[t] for t in terms], dtype=np.float64)
        n = len(texts)
        self.idf_ = np.log((1.0 + n) / (1.0 + doc_freq)) + 1.0
        return self

    def transform(self, texts: Iterable[str]) -> sp.csr_matrix:
        if not self.vocabulary_:
            raise RuntimeError("vectorizer is not fitted")
        indptr = [0]
        indices: list[int] = []
        values: list[float] = []
        vocab = self.vocabulary_
        for text in texts:
            counts = Counter(t for t in ngrams(text, self.ngram_range) if t in vocab)
            for term, count in counts.items():
                indices.append(vocab[term])
                values.append(float(count))
            indptr.append(len(indices))
        X = sp.csr_matrix(
            (np.asarray(values), np.asarray(indices, dtype=np.int64), np.asarray(indptr)),
            shape=(len(indptr) - 1, len(vocab)),
        )
        X.sort_indices()
        if self.sublinear_tf:
            X.data = 1.0 + np.log(X.data)
        X = X @ sp.diags(self.idf_)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        return sp.csr_matrix(sp.diags(1.0 / norms) @ X)

    def fit_transform(self, texts: Sequence[str]) -> sp.csr_matrix:
        return self.fit(texts).transform(texts)
