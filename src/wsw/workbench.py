"""Session state behind the web app: LF columns, human labels, the label model and the queue.

Kept free of UI code so the whole authoring-labelling cycle is testable without a browser.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np

from wsw.active.strategies import Pool, Strategy, make_strategy, posterior_margin
from wsw.data.dataset import Dataset
from wsw.diagnostics import (
    LFStats,
    MatrixSummary,
    PairMatrices,
    conflict_entropy,
    lf_stats,
    matrix_summary,
    pair_matrices,
)
from wsw.endmodel import TextFeatures, train_end_model, training_targets
from wsw.labelmodels import LabelModel, make_label_model
from wsw.labelmodels.dawid_skene import DawidSkene
from wsw.lf import ABSTAIN, LABEL_DTYPE, KeywordLF, LabelingFunction, RegexLF, apply_lfs
from wsw.llm import ChatProvider, LLMLabelingFunction
from wsw.metrics import accuracy, f1


@dataclass(frozen=True)
class LFColumn:
    name: str
    kind: str
    description: str
    train: np.ndarray
    test: np.ndarray


@dataclass(frozen=True)
class QueueItem:
    index: int
    text: str
    votes: list[tuple[str, str]]
    posterior: list[float]
    conflict: float
    margin: float


@dataclass(frozen=True)
class Evaluation:
    label_model_accuracy: float
    label_model_f1: float
    end_model_accuracy: float
    end_model_f1: float
    train_rows: int
    human_labels: int
    seconds: float


class Workbench:
    def __init__(
        self,
        dataset: Dataset,
        *,
        label_model: str = "dawid_skene",
        strategy: str = "conflict_entropy",
        seed: int = 0,
    ) -> None:
        self.dataset = dataset
        self.k = dataset.k
        self.columns: list[LFColumn] = [
            LFColumn(name, "precomputed", "shipped with the dataset", train, test)
            for name, train, test in zip(
                dataset.lf_names, dataset.train.L.T, dataset.test.L.T, strict=True
            )
        ]
        self.gold: dict[int, int] = {}
        self.skipped: set[int] = set()
        self.label_model_name = label_model
        self.strategy_name = strategy
        self._strategy: Strategy = make_strategy(strategy)
        self._rng = np.random.default_rng(seed)
        self._features: TextFeatures | None = None
        self._L_cache: np.ndarray | None = None
        self.model: LabelModel | None = None
        self.posterior = np.full((len(dataset.train), self.k), 1.0 / self.k)
        self.refit()

    @property
    def label_names(self) -> list[str]:
        return self.dataset.label_names

    @property
    def texts(self) -> list[str]:
        return self.dataset.train.texts

    @property
    def lf_names(self) -> list[str]:
        return [c.name for c in self.columns]

    @property
    def L(self) -> np.ndarray:
        if self._L_cache is None:
            self._L_cache = self._stack("train")
        return self._L_cache

    def _stack(self, split: str) -> np.ndarray:
        n = len(getattr(self.dataset, split))
        if not self.columns:
            return np.empty((n, 0), dtype=LABEL_DTYPE)
        return np.column_stack([getattr(c, split) for c in self.columns]).astype(LABEL_DTYPE)

    # LF authoring -------------------------------------------------------------------------

    def add_lf(self, lf: LabelingFunction, kind: str, description: str) -> LFColumn:
        if not lf.name.strip():
            raise ValueError("an LF needs a name")
        if lf.name in self.lf_names:
            raise ValueError(f"an LF called {lf.name!r} already exists")
        column = LFColumn(
            name=lf.name,
            kind=kind,
            description=description,
            train=apply_lfs([lf], self.texts, self.k)[:, 0],
            test=apply_lfs([lf], self.dataset.test.texts, self.k)[:, 0],
        )
        self.columns.append(column)
        self._L_cache = None
        self.refit()
        return column

    def add_keyword_lf(self, name: str, keywords: str, label: int) -> LFColumn:
        words = tuple(w for w in (part.strip() for part in keywords.split(",")) if w)
        lf = KeywordLF(name.strip(), words, self._check_label(label))
        return self.add_lf(lf, "keyword", ", ".join(lf.keywords))

    def add_regex_lf(self, name: str, pattern: str, label: int) -> LFColumn:
        lf = RegexLF(name.strip(), pattern, self._check_label(label))
        return self.add_lf(lf, "regex", pattern)

    def add_llm_lf(self, name: str, instruction: str, provider: ChatProvider) -> LFColumn:
        if not instruction.strip():
            raise ValueError("an LLM labelling function needs an instruction")
        lf = LLMLabelingFunction(name.strip(), provider, tuple(self.label_names), instruction)
        return self.add_lf(lf, "llm", instruction.strip())

    def remove_lf(self, name: str) -> None:
        if name not in self.lf_names:
            raise ValueError(f"no LF called {name!r}")
        self.columns = [c for c in self.columns if c.name != name]
        self._L_cache = None
        self.refit()

    def _check_label(self, label: int) -> int:
        if not 0 <= int(label) < self.k:
            raise ValueError(f"label must be one of 0..{self.k - 1}")
        return int(label)

    # Label model ----------------------------------------------------------------------------

    def set_label_model(self, name: str) -> None:
        self.label_model_name = name
        self.refit()

    def set_strategy(self, name: str) -> None:
        self._strategy = make_strategy(name)
        self.strategy_name = name

    def _class_balance(self) -> np.ndarray | str:
        # The tensor estimator needs three LFs; with fewer, fall back to a uniform prior.
        return "estimate" if self.L.shape[1] >= 3 else "uniform"

    def refit(self) -> None:
        L = self.L
        if L.shape[1] == 0:
            self.model = None
            self.posterior = np.full((L.shape[0], self.k), 1.0 / self.k)
            return
        model = make_label_model(self.label_model_name, self._class_balance())
        if isinstance(model, DawidSkene):
            idx, ys = self._gold_arrays()
            model.fit(L, self.k, gold_idx=idx, gold_y=ys)
        else:
            model.fit(L, self.k)
        self.model = model
        posterior = model.predict_proba(L)
        # A human label is the answer for its row, whatever the model would predict there.
        idx, ys = self._gold_arrays()
        posterior[idx] = np.eye(self.k)[ys]
        self.posterior = posterior

    # Diagnostics ------------------------------------------------------------------------------

    def _gold_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        idx = np.fromiter(self.gold.keys(), dtype=np.int64, count=len(self.gold))
        return idx, np.fromiter(self.gold.values(), dtype=np.int64, count=len(self.gold))

    def stats(self) -> list[LFStats]:
        idx, ys = self._gold_arrays()
        return lf_stats(self.L, self.k, self.lf_names, idx, ys)

    def estimated_accuracy(self) -> np.ndarray | None:
        return None if self.model is None else self.model.lf_accuracy()

    def pairs(self) -> PairMatrices:
        return pair_matrices(self.L, self.k)

    def summary(self) -> MatrixSummary:
        return matrix_summary(self.L, self.k)

    def most_conflicted(self, n: int = 10) -> list[QueueItem]:
        scores = conflict_entropy(self.L, self.k)
        order = np.argsort(-scores, kind="stable")[:n]
        return [self._item(int(i)) for i in order if scores[i] > 0]

    # Queue --------------------------------------------------------------------------------------

    def _features_for_queue(self) -> TextFeatures:
        if self._features is None:
            self._features = TextFeatures.build(self.dataset, min_df=1, max_features=30000)
        return self._features

    def next_item(self) -> QueueItem | None:
        done = np.zeros(len(self.texts), dtype=bool)
        done[list(self.gold)] = True
        done[list(self.skipped)] = True
        if done.all():
            return None
        features = self._features_for_queue().train if self.strategy_name == "coreset" else None
        pool = Pool(self.L, self.k, features)
        picked = self._strategy.select(pool, self.posterior, done, 1, self._rng)
        return self._item(int(picked[0])) if picked.size else None

    def _item(self, i: int) -> QueueItem:
        row = self.L[i]
        votes = [
            (name, self.label_names[int(v)])
            for name, v in zip(self.lf_names, row, strict=True)
            if v != ABSTAIN
        ]
        P = self.posterior[i : i + 1]
        return QueueItem(
            index=i,
            text=self.texts[i],
            votes=votes,
            posterior=[float(p) for p in P[0]],
            conflict=float(conflict_entropy(row[None, :], self.k)[0]) if row.size else 0.0,
            margin=float(posterior_margin(P)[0]),
        )

    def submit(self, index: int, label: int) -> None:
        self.gold[int(index)] = self._check_label(label)
        self.skipped.discard(int(index))
        self.refit()

    def skip(self, index: int) -> None:
        self.skipped.add(int(index))

    # End model and export --------------------------------------------------------------------

    def evaluate(self) -> Evaluation:
        """Train the TF-IDF end model on the current soft labels; score both models on test."""
        started = time.perf_counter()
        features = self._features_for_queue()
        idx, ys = self._gold_arrays()
        rows, targets = training_targets(self.L, self.posterior, idx, ys)
        if rows.size == 0:
            raise ValueError("no training rows yet: add an LF or label a few examples")
        end_model = train_end_model(features.train, rows, targets)
        end_pred = end_model.predict(features.test)
        test_L = self._stack("test")
        test_y = self.dataset.test.y
        if self.model is None:
            lm_pred = self._rng.integers(0, self.k, size=test_y.shape[0])
        else:
            lm_pred = self.model.predict(test_L, self._rng)
        return Evaluation(
            label_model_accuracy=accuracy(test_y, lm_pred),
            label_model_f1=f1(test_y, lm_pred, self.k),
            end_model_accuracy=accuracy(test_y, end_pred),
            end_model_f1=f1(test_y, end_pred, self.k),
            train_rows=int(rows.size),
            human_labels=len(self.gold),
            seconds=time.perf_counter() - started,
        )

    def export_jsonl(self) -> str:
        """One JSON object per covered or human-labelled row: text, label and its source."""
        covered = (self.L != ABSTAIN).any(axis=1)
        lines: list[str] = []
        for i, text in enumerate(self.texts):
            if i in self.gold:
                label, source, confidence = self.gold[i], "human", 1.0
            elif covered[i]:
                label = int(np.argmax(self.posterior[i]))
                source, confidence = self.label_model_name, float(self.posterior[i, label])
            else:
                continue
            record = {
                "row": i,
                "text": text,
                "label": self.label_names[label],
                "source": source,
                "confidence": round(confidence, 4),
            }
            lines.append(json.dumps(record, ensure_ascii=False))
        return "\n".join(lines) + ("\n" if lines else "")
