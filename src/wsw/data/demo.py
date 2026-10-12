"""A small synthetic corpus (app-store reviews) for the UI demo, tests and the Docker smoke run.

It is generated from word lists with a fixed seed, so it needs no download and has known
structure: three classes, keyword LFs with different precision, and deliberate conflicts
("would love" is a feature request, "love" alone is praise).
"""

from __future__ import annotations

import numpy as np

from wsw.data.dataset import Dataset, Split
from wsw.lf import KeywordLF, LabelingFunction, apply_lfs

LABELS = ["bug", "feature", "praise"]
CLASS_BALANCE = (0.45, 0.30, 0.25)

_CUES: dict[int, list[str]] = {
    0: ["crashes", "freezes", "error", "fails", "broken", "slow", "stuck", "keeps closing"],
    1: ["please add", "would love", "wish", "missing", "an option for", "support for", "request"],
    2: ["love", "great", "awesome", "amazing", "perfect", "best", "thanks"],
}
_TOPICS = [
    "the update",
    "login",
    "sync",
    "export",
    "dark mode",
    "the widget",
    "notifications",
    "battery use",
    "offline mode",
    "the search bar",
    "photo upload",
    "the calendar view",
]
_FILLERS = ["honestly", "since last week", "on my phone", "every time", "for me", "today"]


def demo_lfs() -> list[LabelingFunction]:
    return [
        KeywordLF("kw_crash", ("crashes", "freezes", "keeps closing"), 0),
        KeywordLF("kw_error", ("error", "fails", "broken"), 0),
        KeywordLF("kw_slow", ("slow", "stuck"), 0),
        KeywordLF("kw_update", ("update",), 0),
        KeywordLF("kw_please_add", ("please add", "would love"), 1),
        KeywordLF("kw_wish", ("wish", "missing", "option"), 1),
        KeywordLF("kw_love", ("love", "awesome", "amazing"), 2),
        KeywordLF("kw_great", ("great", "perfect", "best"), 2),
    ]


def _sentence(rng: np.random.Generator, label: int) -> str:
    n_cues = int(rng.integers(1, 3))
    cues = list(rng.choice(_CUES[label], size=n_cues, replace=False))
    if rng.random() < 0.18:
        # Off-class cue: the source of LF conflicts the queue should surface.
        other = int(rng.choice([c for c in _CUES if c != label]))
        cues.append(str(rng.choice(_CUES[other])))
    if rng.random() < 0.12:
        cues = []  # uncovered review: no LF fires unless a topic word does
    words = [*cues, str(rng.choice(_TOPICS)), str(rng.choice(_FILLERS))]
    rng.shuffle(words)
    return " ".join(words).capitalize() + "."


def make_split(rng: np.random.Generator, n: int, lfs: list[LabelingFunction]) -> Split:
    y = rng.choice(len(LABELS), size=n, p=CLASS_BALANCE)
    texts = [_sentence(rng, int(label)) for label in y]
    return Split(texts, apply_lfs(lfs, texts, len(LABELS)), y.astype(np.int64))


def make_demo_dataset(
    n_train: int = 1200, n_valid: int = 200, n_test: int = 400, seed: int = 0
) -> Dataset:
    rng = np.random.default_rng(seed)
    lfs = demo_lfs()
    return Dataset(
        name="demo",
        label_names=list(LABELS),
        lf_names=[lf.name for lf in lfs],
        train=make_split(rng, n_train, lfs),
        valid=make_split(rng, n_valid, lfs),
        test=make_split(rng, n_test, lfs),
        notes={"source": f"synthetic app-store reviews, seed={seed}"},
    )
