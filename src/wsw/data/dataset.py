"""In-memory dataset representation shared by the label models, the benchmark and the UI."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wsw.lf import ABSTAIN


@dataclass(frozen=True)
class Split:
    """One split: raw texts, the precomputed label matrix and gold labels."""

    texts: list[str]
    L: np.ndarray
    y: np.ndarray

    def __post_init__(self) -> None:
        if self.L.ndim != 2:
            raise ValueError(f"label matrix must be 2-D, got shape {self.L.shape}")
        if not (len(self.texts) == self.L.shape[0] == self.y.shape[0]):
            raise ValueError("texts, L and y must have the same number of rows")
        if self.L.size and self.L.min() < ABSTAIN:
            raise ValueError("label matrix entries must be >= -1 (abstain)")

    def __len__(self) -> int:
        return len(self.texts)

    def subset(self, idx: np.ndarray) -> Split:
        return Split([self.texts[i] for i in idx], self.L[idx], self.y[idx])


@dataclass(frozen=True)
class Dataset:
    """A classification task with weak labels on train/valid/test."""

    name: str
    label_names: list[str]
    lf_names: list[str]
    train: Split
    valid: Split
    test: Split
    notes: dict[str, str] = field(default_factory=dict)

    @property
    def k(self) -> int:
        return len(self.label_names)

    @property
    def m(self) -> int:
        return self.train.L.shape[1]

    def __post_init__(self) -> None:
        for split in (self.train, self.valid, self.test):
            if split.L.shape[1] != len(self.lf_names):
                raise ValueError("every split must have one column per labelling function")
            if split.L.size and split.L.max() >= self.k:
                raise ValueError("label matrix contains a class id outside the label set")
