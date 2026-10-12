from __future__ import annotations

import os

import numpy as np
import pytest
from hypothesis import settings

from wsw.data import wrench
from wsw.data.synthetic import SyntheticLabels, sample_label_matrix

settings.register_profile("default", deadline=None, max_examples=60)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))


@pytest.fixture(scope="session")
def binary_synthetic() -> SyntheticLabels:
    return sample_label_matrix(
        n=20000,
        accuracy=np.array([0.85, 0.75, 0.65, 0.9, 0.7, 0.8]),
        coverage=np.array([0.5, 0.4, 0.6, 0.3, 0.45, 0.35]),
        balance=np.array([0.7, 0.3]),
        seed=11,
    )


@pytest.fixture(scope="session")
def multiclass_synthetic() -> SyntheticLabels:
    return sample_label_matrix(
        n=30000,
        accuracy=np.array([0.8, 0.7, 0.75, 0.85, 0.65, 0.9, 0.7]),
        coverage=np.array([0.5, 0.6, 0.4, 0.3, 0.55, 0.35, 0.45]),
        balance=np.array([0.5, 0.3, 0.2]),
        seed=5,
    )


def wrench_present() -> bool:
    return all(wrench.is_available(name) for name in wrench.WRENCH_DATASETS)


requires_wrench = pytest.mark.skipif(
    not wrench_present(), reason="WRENCH data not downloaded (run `wsw fetch all`)"
)
