"""Classification metrics, written out so the benchmark does not depend on scikit-learn.

``f1`` follows WRENCH's convention: binary F1 of class 1 for two-class tasks (spam on SMS
and YouTube, positive on Yelp) and macro-F1 otherwise.
"""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("y_true and y_pred must be non-empty and of equal shape")
    return float((y_true == y_pred).mean())


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> np.ndarray:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    confusion = np.zeros((k, k), dtype=np.int64)
    np.add.at(confusion, (y_true, y_pred), 1)
    tp = np.diag(confusion).astype(np.float64)
    denom = confusion.sum(axis=0) + confusion.sum(axis=1)
    # A class that is neither present nor predicted gets F1 = 0, as in scikit-learn.
    return np.divide(2.0 * tp, denom, out=np.zeros(k), where=denom > 0)


def f1(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    scores = per_class_f1(y_true, y_pred, k)
    return float(scores[1]) if k == 2 else float(scores.mean())


def normalised_aulc(budgets: np.ndarray, values: np.ndarray) -> float:
    """Trapezoidal area under a learning curve divided by the budget range.

    The result is the curve's average height on a linear label axis, so it lies in the
    metric's own range and a flat curve at ``v`` scores exactly ``v``.
    """
    x = np.asarray(budgets, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or x.shape != v.shape or x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("need at least two strictly increasing budgets with one value each")
    return float(np.trapezoid(v, x) / (x[-1] - x[0]))
