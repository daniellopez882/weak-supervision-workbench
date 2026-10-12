"""The benchmark behind ``wsw bench``: label models, end models, active-learning curves and
parameter recovery on synthetic data. Results go to one JSON file, rewritten after every
dataset so an interrupted run keeps what it finished.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from wsw.active import LoopConfig, SimulatedOracle, Task, run_loop
from wsw.benchmark.machine import git_state, local_now, machine_info
from wsw.data import load_dataset
from wsw.data.dataset import Dataset
from wsw.data.synthetic import random_lf_parameters, sample_label_matrix
from wsw.endmodel import TextFeatures, train_end_model, training_targets
from wsw.labelmodels import (
    DawidSkene,
    FlyingSquid,
    LabelModel,
    MajorityVote,
    MeTaL,
    balance_from_labels,
    estimate_class_balance,
)
from wsw.labelmodels.base import argmax_random_ties
from wsw.lf import ABSTAIN
from wsw.metrics import accuracy, f1, normalised_aulc

Log = Callable[[str], None]


@dataclass(frozen=True)
class SuiteConfig:
    name: str
    datasets: tuple[str, ...]
    seeds: tuple[int, ...]
    budgets: tuple[int, ...]
    batch_size: int
    al_pool_cap: int | None
    strategies: tuple[str, ...] = ("random", "conflict_entropy", "posterior_margin", "coreset")
    loop_label_model: str = "dawid_skene"
    end_model_C: float = 1.0
    tfidf: dict[str, Any] = field(
        default_factory=lambda: {"ngram_range": (1, 2), "min_df": 2, "max_features": 30000}
    )
    snorkel_epochs: int = 100
    synthetic_n: int = 20000
    synthetic_m: int = 10


SUITES: dict[str, SuiteConfig] = {
    "full": SuiteConfig(
        name="full",
        datasets=("youtube", "sms", "trec", "yelp", "agnews"),
        seeds=(0, 1, 2, 3, 4),
        budgets=(25, 50, 100, 250, 500),
        batch_size=25,
        al_pool_cap=10000,
    ),
    "smoke": SuiteConfig(
        name="smoke",
        datasets=("demo",),
        seeds=(0, 1),
        budgets=(10, 20, 40),
        batch_size=10,
        al_pool_cap=600,
        tfidf={"ngram_range": (1, 2), "min_df": 1, "max_features": 30000},
        snorkel_epochs=100,
        synthetic_n=3000,
        synthetic_m=6,
    ),
}


def _mean_std(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0}


class SnorkelReference:
    """Snorkel's LabelModel, used only as a reference point (optional dependency group)."""

    name = "snorkel"

    def __init__(self, class_balance: np.ndarray, seed: int, epochs: int) -> None:
        from snorkel.labeling.model import LabelModel as SnorkelLabelModel

        self._cls = SnorkelLabelModel
        self.class_balance, self.seed, self.epochs = class_balance, seed, epochs
        self.k: int | None = None
        self.model: Any = None

    def fit(self, L: np.ndarray, k: int) -> SnorkelReference:
        self.k = k
        self.model = self._cls(cardinality=k, verbose=False, device="cpu")
        self.model.fit(
            L.astype(np.int64),
            class_balance=self.class_balance,
            n_epochs=self.epochs,
            seed=self.seed,
            progress_bar=False,
        )
        return self

    def predict_proba(self, L: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict_proba(L.astype(np.int64)), dtype=np.float64)

    def lf_accuracy(self) -> np.ndarray:
        assert self.k is not None
        cond = self.model.get_conditional_probs()  # (m, k + 1, k); outcome 0 is abstain
        k, p = self.k, np.asarray(self.class_balance)
        correct = (cond[:, 1 + np.arange(k), np.arange(k)] * p).sum(axis=1)
        votes = ((1.0 - cond[:, 0, :]) * p).sum(axis=1)
        return correct / votes


def snorkel_available() -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import snorkel.labeling.model  # noqa: F401
    except ImportError:
        return False
    return True


# Label models and end models --------------------------------------------------------------------


def _label_model_factories(
    valid_balance: np.ndarray, estimated: np.ndarray, use_snorkel: bool, epochs: int
) -> dict[str, Callable[[int], LabelModel | SnorkelReference]]:
    factories: dict[str, Callable[[int], LabelModel | SnorkelReference]] = {
        "majority_vote": lambda _s: MajorityVote(),
        "dawid_skene": lambda _s: DawidSkene(),
        "metal": lambda _s: MeTaL(class_balance=valid_balance),
        "metal_estimated_balance": lambda _s: MeTaL(class_balance=estimated),
        "flyingsquid": lambda _s: FlyingSquid(class_balance=valid_balance),
        "flyingsquid_estimated_balance": lambda _s: FlyingSquid(class_balance=estimated),
    }
    if use_snorkel:
        factories["snorkel"] = lambda s: SnorkelReference(valid_balance, s, epochs)
    return factories


def _bench_label_models(
    ds: Dataset,
    feats: TextFeatures,
    suite: SuiteConfig,
    use_snorkel: bool,
    log: Log,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    k = ds.k
    valid_balance = balance_from_labels(ds.valid.y, k)
    t0 = time.perf_counter()
    estimate = estimate_class_balance(ds.train.L, k)
    estimate_seconds = time.perf_counter() - t0
    true_balance = np.bincount(ds.train.y, minlength=k) / len(ds.train)
    balance = {
        "true_train": true_balance.tolist(),
        "from_valid_labels": valid_balance.tolist(),
        "estimated": estimate.balance.tolist(),
        "estimated_l1_error": float(np.abs(estimate.balance - true_balance).sum()),
        "valid_l1_error": float(np.abs(valid_balance - true_balance).sum()),
        "estimation_seconds": estimate_seconds,
        "estimation_converged": estimate.converged,
    }

    end_cache: dict[str, tuple[float, float, float, int]] = {}
    out: dict[str, Any] = {}
    factories = _label_model_factories(
        valid_balance, estimate.balance, use_snorkel, suite.snorkel_epochs
    )
    for name, factory in factories.items():
        per_seed: list[dict[str, float]] = []
        for seed in suite.seeds:
            rng = np.random.default_rng([seed, 3])
            model = factory(seed)
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(ds.train.L, k)
            fit_seconds = time.perf_counter() - t0
            test_pred = argmax_random_ties(model.predict_proba(ds.test.L), rng)
            posterior = model.predict_proba(ds.train.L)
            rows, targets = training_targets(ds.train.L, posterior)
            # Deterministic label models give identical soft labels for every seed; the end
            # model is deterministic too, so its result is reused instead of refitted.
            key = hashlib.sha1(np.ascontiguousarray(targets).tobytes()).hexdigest()
            if key not in end_cache:
                t0 = time.perf_counter()
                em = train_end_model(feats.train, rows, targets, C=suite.end_model_C)
                end_seconds = time.perf_counter() - t0
                pred = em.predict(feats.test)
                end_cache[key] = (
                    accuracy(ds.test.y, pred),
                    f1(ds.test.y, pred, k),
                    end_seconds,
                    int(rows.size),
                )
            end_acc, end_f1, end_seconds, n_rows = end_cache[key]
            per_seed.append(
                {
                    "seed": seed,
                    "label_model_accuracy": accuracy(ds.test.y, test_pred),
                    "label_model_f1": f1(ds.test.y, test_pred, k),
                    "fit_seconds": fit_seconds,
                    "end_model_accuracy": end_acc,
                    "end_model_f1": end_f1,
                    "end_model_fit_seconds": end_seconds,
                    "end_model_train_rows": n_rows,
                }
            )
        summary = {
            metric: _mean_std([r[metric] for r in per_seed])
            for metric in (
                "label_model_accuracy",
                "label_model_f1",
                "fit_seconds",
                "end_model_accuracy",
                "end_model_f1",
            )
        }
        out[name] = {**summary, "per_seed": per_seed}
        log(
            f"  {name:30s} lm acc {summary['label_model_accuracy']['mean']:.3f} "
            f"end acc {summary['end_model_accuracy']['mean']:.3f} "
            f"fit {summary['fit_seconds']['mean']:.2f}s"
        )
    return out, balance, valid_balance


# Active learning ------------------------------------------------------------------------------


def _bench_active(
    ds: Dataset, feats: TextFeatures, valid_balance: np.ndarray, suite: SuiteConfig, log: Log
) -> dict[str, Any]:
    n = len(ds.train)
    runs: dict[str, list[dict[str, Any]]] = {}
    variants = [(s, False) for s in suite.strategies] + [("random", True)]
    for seed in suite.seeds:
        if suite.al_pool_cap is not None and n > suite.al_pool_cap:
            pool_idx = np.sort(
                np.random.default_rng([seed, 2]).choice(n, size=suite.al_pool_cap, replace=False)
            )
        else:
            pool_idx = np.arange(n)
        task = Task(
            k=ds.k,
            pool_L=ds.train.L[pool_idx],
            pool_X=feats.train[pool_idx],
            test_L=ds.test.L,
            test_X=feats.test,
            test_y=ds.test.y,
            class_balance=valid_balance,
        )
        for strategy, gold_only in variants:
            config = LoopConfig(
                strategy=strategy,
                seed=seed,
                budgets=suite.budgets,
                batch_size=suite.batch_size,
                label_model=suite.loop_label_model,
                C=suite.end_model_C,
                gold_only=gold_only,
            )
            result = run_loop(task, SimulatedOracle(ds.train.y[pool_idx]), config)
            key = "random_gold_labels_only" if gold_only else strategy
            runs.setdefault(key, []).append(
                {
                    "seed": seed,
                    "pool_size": int(pool_idx.size),
                    "seconds": result.seconds,
                    "points": [asdict(p) for p in result.points],
                }
            )
        log(f"  active learning seed {seed} done")

    out: dict[str, Any] = {}
    for key, seed_runs in runs.items():
        budgets = [p["n_labels"] for p in seed_runs[0]["points"]]
        curve: dict[str, Any] = {"n_labels": budgets}
        metrics = ["end_accuracy", "end_f1"]
        if key != "random_gold_labels_only":
            metrics += ["lm_accuracy", "lm_f1"]
        for metric in metrics:
            values = np.asarray([[p[metric] for p in r["points"]] for r in seed_runs])
            curve[metric] = {
                "mean": values.mean(axis=0).tolist(),
                "std": (
                    values.std(axis=0, ddof=1) if len(seed_runs) > 1 else 0 * values[0]
                ).tolist(),
            }
        aulc: dict[str, Any] = {}
        for metric in ("end_accuracy", "end_f1"):
            per_seed = []
            for r in seed_runs:
                pts = [p for p in r["points"] if p["n_labels"] > 0]
                per_seed.append(
                    normalised_aulc(
                        np.asarray([p["n_labels"] for p in pts]),
                        np.asarray([p[metric] for p in pts]),
                    )
                )
            aulc[metric] = {**_mean_std(per_seed), "per_seed": per_seed}
        out[key] = {
            "curve": curve,
            "nAULC": aulc,
            "seconds": _mean_std([r["seconds"] for r in seed_runs]),
            "pool_size": seed_runs[0]["pool_size"],
            "runs": seed_runs,
        }
    # Paired against random: same seed, same pool, same starting label model.
    base = out["random"]["nAULC"]["end_accuracy"]["per_seed"]
    for entry in out.values():
        diffs = [
            a - b for a, b in zip(entry["nAULC"]["end_accuracy"]["per_seed"], base, strict=True)
        ]
        entry["nAULC_accuracy_minus_random"] = {
            **_mean_std(diffs),
            "seeds_better": int(sum(d > 0 for d in diffs)),
            "seeds": len(diffs),
        }
    for key in suite.strategies:
        a = out[key]["nAULC"]["end_accuracy"]
        log(f"  {key:30s} nAULC acc {a['mean']:.3f} +- {a['std']:.3f}")
    return out


# Synthetic parameter recovery ------------------------------------------------------------------


def _bench_synthetic(suite: SuiteConfig, use_snorkel: bool, log: Log) -> dict[str, Any]:
    settings = {"k2": np.array([0.7, 0.3]), "k4": np.array([0.4, 0.3, 0.2, 0.1])}
    out: dict[str, Any] = {}
    for label, balance in settings.items():
        k = balance.shape[0]
        errors: dict[str, list[float]] = {}
        seconds: dict[str, list[float]] = {}
        balance_err: list[float] = []
        ds_prior_err: list[float] = []
        for seed in suite.seeds:
            acc, cov = random_lf_parameters(
                suite.synthetic_m, seed, accuracy=(0.55, 0.9), coverage=(0.1, 0.6)
            )
            data = sample_label_matrix(suite.synthetic_n, acc, cov, balance, seed)
            models: dict[str, LabelModel | SnorkelReference] = {
                "dawid_skene": DawidSkene(),
                "metal": MeTaL(class_balance=balance),
            }
            if k == 2:
                models["flyingsquid"] = FlyingSquid(class_balance=balance)
            if use_snorkel:
                models["snorkel"] = SnorkelReference(balance, seed, suite.snorkel_epochs)
            for name, model in models.items():
                t0 = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(data.L, k)
                seconds.setdefault(name, []).append(time.perf_counter() - t0)
                est = model.lf_accuracy()
                assert est is not None
                errors.setdefault(name, []).append(float(np.nanmean(np.abs(est - acc))))
                if isinstance(model, DawidSkene):
                    assert model.class_prior_ is not None
                    ds_prior_err.append(float(np.abs(model.class_prior_ - balance).sum()))
            est_balance = estimate_class_balance(data.L, k).balance
            balance_err.append(float(np.abs(est_balance - balance).sum()))
        out[label] = {
            "n": suite.synthetic_n,
            "m": suite.synthetic_m,
            "balance": balance.tolist(),
            "lf_accuracy_mae": {name: _mean_std(v) for name, v in errors.items()},
            "fit_seconds": {name: _mean_std(v) for name, v in seconds.items()},
            "tensor_balance_l1_error": _mean_std(balance_err),
            "dawid_skene_prior_l1_error": _mean_std(ds_prior_err),
        }
        log(
            f"  synthetic {label}: "
            + ", ".join(f"{name} MAE {np.mean(v):.4f}" for name, v in errors.items())
        )
    return out


# Driver ----------------------------------------------------------------------------------------


def _dataset_info(ds: Dataset) -> dict[str, Any]:
    return {
        "k": ds.k,
        "labels": ds.label_names,
        "n_lfs": ds.m,
        "rows": {"train": len(ds.train), "valid": len(ds.valid), "test": len(ds.test)},
        "train_coverage": float((ds.train.L != ABSTAIN).any(axis=1).mean()),
        "source": ds.notes.get("source", ""),
    }


def run_suite(
    suite: SuiteConfig,
    out_path: Path,
    command: str,
    *,
    use_snorkel: bool = True,
    log: Log = lambda msg: print(msg, file=sys.stderr, flush=True),
) -> dict[str, Any]:
    started = time.perf_counter()
    now = local_now()
    snorkel_ok = use_snorkel and snorkel_available()
    result: dict[str, Any] = {
        "suite": suite.name,
        "date": now.date().isoformat(),
        "timestamp": now.isoformat(timespec="seconds"),
        **git_state(),
        "command": command,
        "machine": machine_info(),
        "config": asdict(suite),
        "snorkel_reference": "run" if snorkel_ok else "not run (snorkel not installed or disabled)",
        "llm_labelling_function": "not run",
        "datasets": {},
        "label_models": {},
        "class_balance": {},
        "end_model_features": {},
        "active_learning": {},
        "synthetic_recovery": {},
    }

    def save() -> None:
        result["wall_seconds"] = time.perf_counter() - started
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    for name in suite.datasets:
        t_ds = time.perf_counter()
        log(f"[{name}] loading")
        ds = load_dataset(name)
        result["datasets"][name] = _dataset_info(ds)
        t0 = time.perf_counter()
        feats = TextFeatures.build(ds, **suite.tfidf)
        result["end_model_features"][name] = {
            "vocabulary": int(feats.train.shape[1]),
            "seconds": time.perf_counter() - t0,
        }
        log(f"[{name}] label models")
        lm, balance, valid_balance = _bench_label_models(ds, feats, suite, snorkel_ok, log)
        result["label_models"][name] = lm
        result["class_balance"][name] = balance
        save()
        log(f"[{name}] active learning")
        result["active_learning"][name] = _bench_active(ds, feats, valid_balance, suite, log)
        result["datasets"][name]["seconds"] = time.perf_counter() - t_ds
        save()

    log("[synthetic] parameter recovery")
    result["synthetic_recovery"] = _bench_synthetic(suite, snorkel_ok, log)
    save()
    log(f"done in {result['wall_seconds']:.0f}s -> {out_path}")
    return result
