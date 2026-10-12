"""From-scratch label models and a name-based factory."""

from __future__ import annotations

from collections.abc import Callable

from wsw.labelmodels.base import LabelModel, argmax_random_ties
from wsw.labelmodels.class_balance import (
    ClassBalanceEstimate,
    balance_from_labels,
    estimate_class_balance,
)
from wsw.labelmodels.dawid_skene import DawidSkene
from wsw.labelmodels.flyingsquid import FlyingSquid
from wsw.labelmodels.majority import MajorityVote
from wsw.labelmodels.metal import BalanceSpec, MeTaL

LABEL_MODELS: dict[str, Callable[[BalanceSpec], LabelModel]] = {
    "majority_vote": lambda _balance: MajorityVote(),
    "dawid_skene": lambda _balance: DawidSkene(),
    "metal": lambda balance: MeTaL(class_balance=balance),
    "flyingsquid": lambda balance: FlyingSquid(class_balance=balance),
}


def make_label_model(name: str, class_balance: BalanceSpec = "estimate") -> LabelModel:
    """Build a label model by name; ``class_balance`` only matters for MeTaL and FlyingSquid
    (Dawid-Skene estimates its prior by EM, majority vote has none)."""
    try:
        factory = LABEL_MODELS[name]
    except KeyError:
        raise ValueError(
            f"unknown label model {name!r}; choose from {sorted(LABEL_MODELS)}"
        ) from None
    return factory(class_balance)


__all__ = [
    "LABEL_MODELS",
    "BalanceSpec",
    "ClassBalanceEstimate",
    "DawidSkene",
    "FlyingSquid",
    "LabelModel",
    "MajorityVote",
    "MeTaL",
    "argmax_random_ties",
    "balance_from_labels",
    "estimate_class_balance",
    "make_label_model",
]
