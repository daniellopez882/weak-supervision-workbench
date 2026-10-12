"""Name-based dataset lookup used by the CLI, the benchmark and the app."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from wsw.data import wrench
from wsw.data.dataset import Dataset
from wsw.data.demo import make_demo_dataset


def available_datasets() -> tuple[str, ...]:
    return ("demo", *wrench.WRENCH_DATASETS)


def downloaded_datasets(root: Path | None = None) -> tuple[str, ...]:
    """The demo corpus plus every WRENCH dataset whose files are present."""
    return ("demo", *(n for n in wrench.WRENCH_DATASETS if wrench.is_available(n, root)))


def load_dataset(name: str, root: Path | None = None) -> Dataset:
    if name == "demo":
        return make_demo_dataset()
    if name in wrench.WRENCH_DATASETS:
        return wrench.load(name, root)
    raise ValueError(f"unknown dataset {name!r}; choose from {available_datasets()}")


def cap_train_rows(dataset: Dataset, max_rows: int | None, seed: int = 0) -> Dataset:
    """Keep a seeded random subset of the training split (valid and test stay whole)."""
    n = len(dataset.train)
    if max_rows is None or n <= max_rows:
        return dataset
    idx = np.sort(np.random.default_rng(seed).choice(n, size=max_rows, replace=False))
    notes = {**dataset.notes, "train_rows": f"{max_rows} of {n} (seed {seed})"}
    return dataclasses.replace(dataset, train=dataset.train.subset(idx), notes=notes)
