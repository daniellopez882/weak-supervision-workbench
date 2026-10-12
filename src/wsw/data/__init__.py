"""Datasets: WRENCH loaders/fetcher, the demo corpus and synthetic label matrices."""

from wsw.data.dataset import Dataset, Split
from wsw.data.registry import (
    available_datasets,
    cap_train_rows,
    downloaded_datasets,
    load_dataset,
)

__all__ = [
    "Dataset",
    "Split",
    "available_datasets",
    "cap_train_rows",
    "downloaded_datasets",
    "load_dataset",
]
