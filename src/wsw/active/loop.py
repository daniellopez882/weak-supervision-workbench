"""Seeded active-learning loop with a simulated oracle.

The loop starts from pure weak supervision (0 human labels), asks a strategy for batches of
rows, reveals their gold labels through the oracle, refits the label model with those labels
clamped (when the model supports it) and retrains the end model at each budget checkpoint.
All randomness comes from generators derived from ``config.seed``, so two runs with the same
inputs and seed select the same rows and report the same numbers.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from wsw.active.strategies import Pool, make_strategy
from wsw.endmodel import SoftLabelLogisticRegression, train_end_model, training_targets
from wsw.labelmodels import LabelModel, make_label_model
from wsw.labelmodels.dawid_skene import DawidSkene
from wsw.metrics import accuracy, f1

DEFAULT_BUDGETS = (25, 50, 100, 250, 500)


class SimulatedOracle:
    """Answers label queries from the gold labels and counts how many it was asked."""

    def __init__(self, y: np.ndarray) -> None:
        self._y = np.asarray(y, dtype=np.int64)
        self.queries = 0

    def __call__(self, idx: np.ndarray) -> np.ndarray:
        self.queries += int(idx.size)
        return self._y[idx]


@dataclass(frozen=True)
class Task:
    """Everything that stays fixed across strategies and seeds for one dataset."""

    k: int
    pool_L: np.ndarray
    pool_X: sp.csr_matrix
    test_L: np.ndarray
    test_X: sp.csr_matrix
    test_y: np.ndarray
    class_balance: np.ndarray | str = "estimate"


@dataclass(frozen=True)
class LoopConfig:
    strategy: str
    seed: int = 0
    budgets: tuple[int, ...] = DEFAULT_BUDGETS
    batch_size: int = 25
    label_model: str = "dawid_skene"
    C: float = 1.0
    #: Train the end model on the human labels only (no weak labels), as a reference curve.
    gold_only: bool = False

    def __post_init__(self) -> None:
        if not self.budgets or any(b <= 0 for b in self.budgets):
            raise ValueError("budgets must be positive")
        if list(self.budgets) != sorted(set(self.budgets)):
            raise ValueError("budgets must be strictly increasing")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")


@dataclass(frozen=True)
class CurvePoint:
    n_labels: int
    end_accuracy: float
    end_f1: float
    lm_accuracy: float
    lm_f1: float


@dataclass(frozen=True)
class LoopResult:
    config: LoopConfig
    points: tuple[CurvePoint, ...]
    selected: np.ndarray
    seconds: float
    queries: int


def _fit_label_model(
    config: LoopConfig, task: Task, gold_idx: np.ndarray, gold_y: np.ndarray
) -> LabelModel:
    model = make_label_model(config.label_model, task.class_balance)
    if isinstance(model, DawidSkene):
        return model.fit(task.pool_L, task.k, gold_idx=gold_idx, gold_y=gold_y)
    return model.fit(task.pool_L, task.k)


def run_loop(task: Task, oracle: SimulatedOracle, config: LoopConfig) -> LoopResult:
    started = time.perf_counter()
    select_rng = np.random.default_rng([config.seed, 0])
    eval_rng = np.random.default_rng([config.seed, 1])
    strategy = make_strategy(config.strategy)
    pool = Pool(task.pool_L, task.k, task.pool_X)
    n = task.pool_L.shape[0]
    labelled = np.zeros(n, dtype=bool)
    order: list[int] = []
    answers: list[int] = []

    model = _fit_label_model(config, task, np.empty(0, np.int64), np.empty(0, np.int64))
    posterior = model.predict_proba(task.pool_L)
    end_model: SoftLabelLogisticRegression | None = None
    checkpoints = config.budgets if config.gold_only else (0, *config.budgets)
    points: list[CurvePoint] = []

    for target in checkpoints:
        while len(order) < min(target, n):
            want = min(config.batch_size, target - len(order))
            picked = strategy.select(pool, posterior, labelled, want, select_rng)
            if picked.size == 0:
                break
            labelled[picked] = True
            order.extend(int(i) for i in picked)
            answers.extend(int(a) for a in oracle(picked))
            if model.accepts_gold:
                model = _fit_label_model(config, task, np.asarray(order), np.asarray(answers))
                posterior = model.predict_proba(task.pool_L)

        gold_idx, gold_y = np.asarray(order, dtype=np.int64), np.asarray(answers, dtype=np.int64)
        if config.gold_only:
            rows, targets = gold_idx, np.eye(task.k)[gold_y]
        else:
            rows, targets = training_targets(task.pool_L, posterior, gold_idx, gold_y)
        end_model = train_end_model(task.pool_X, rows, targets, C=config.C, warm_start=end_model)
        end_pred = end_model.predict(task.test_X)
        lm_pred = model.predict(task.test_L, eval_rng)
        points.append(
            CurvePoint(
                n_labels=len(order),
                end_accuracy=accuracy(task.test_y, end_pred),
                end_f1=f1(task.test_y, end_pred, task.k),
                lm_accuracy=accuracy(task.test_y, lm_pred),
                lm_f1=f1(task.test_y, lm_pred, task.k),
            )
        )

    return LoopResult(
        config=config,
        points=tuple(points),
        selected=np.asarray(order, dtype=np.int64),
        seconds=time.perf_counter() - started,
        queries=oracle.queries,
    )
