from __future__ import annotations

import json

import numpy as np
import pytest

from wsw.data import load_dataset
from wsw.data.dataset import Dataset
from wsw.lf import ABSTAIN
from wsw.llm import KeywordFakeProvider
from wsw.workbench import Workbench


@pytest.fixture(scope="module")
def demo() -> Dataset:
    return load_dataset("demo")


def test_adding_an_lf_extends_the_matrix_and_refits(demo: Dataset) -> None:
    wb = Workbench(demo)
    before = wb.L.shape[1]
    column = wb.add_keyword_lf("kw_sync", "sync, export", 0)
    assert wb.L.shape[1] == before + 1
    assert column.kind == "keyword"
    assert column.description == "sync, export"
    stats = {s.name: s for s in wb.stats()}
    assert 0 < stats["kw_sync"].coverage < 1
    assert wb.posterior.shape == (len(demo.train), demo.k)
    with pytest.raises(ValueError, match="already exists"):
        wb.add_keyword_lf("kw_sync", "login", 1)
    with pytest.raises(ValueError, match="label must be"):
        wb.add_regex_lf("re_bad", "x", 7)
    wb.remove_lf("kw_sync")
    assert wb.L.shape[1] == before


def test_queue_never_repeats_labelled_or_skipped_rows(demo: Dataset) -> None:
    wb = Workbench(demo, strategy="conflict_entropy", seed=1)
    first = wb.next_item()
    assert first is not None
    assert first.conflict > 0  # the demo has conflicting LFs, so the top row has a split vote
    wb.submit(first.index, int(demo.train.y[first.index]))
    second = wb.next_item()
    assert second is not None and second.index != first.index
    wb.skip(second.index)
    third = wb.next_item()
    assert third is not None and third.index not in {first.index, second.index}
    assert wb.gold == {first.index: int(demo.train.y[first.index])}


def test_human_labels_are_clamped_and_show_up_in_lf_accuracy(demo: Dataset) -> None:
    wb = Workbench(demo)
    rows = np.flatnonzero((demo.train.L != ABSTAIN).any(axis=1))[:5]
    for i in rows:
        wb.submit(int(i), 2)
    np.testing.assert_allclose(wb.posterior[rows], np.tile([0, 0, 1], (5, 1)), atol=1e-9)
    assert any(s.n_labelled_votes > 0 for s in wb.stats())


def test_label_model_and_strategy_switch(demo: Dataset) -> None:
    wb = Workbench(demo)
    for name in ("majority_vote", "metal", "flyingsquid", "dawid_skene"):
        wb.set_label_model(name)
        np.testing.assert_allclose(wb.posterior.sum(axis=1), 1.0)
    wb.set_strategy("coreset")
    assert wb.next_item() is not None
    acc = wb.estimated_accuracy()
    assert acc is not None and acc.shape == (wb.L.shape[1],)


def test_removing_every_lf_leaves_a_uniform_posterior(demo: Dataset) -> None:
    wb = Workbench(demo)
    for name in list(wb.lf_names):
        wb.remove_lf(name)
    assert wb.model is None
    np.testing.assert_allclose(wb.posterior, 1 / demo.k)
    assert wb.export_jsonl() == ""


def test_evaluate_and_export(demo: Dataset) -> None:
    wb = Workbench(demo)
    wb.submit(0, int(demo.train.y[0]))
    ev = wb.evaluate()
    assert 0.5 < ev.end_model_accuracy <= 1
    assert ev.human_labels == 1
    lines = [json.loads(line) for line in wb.export_jsonl().splitlines()]
    assert lines[0]["row"] == 0 and lines[0]["source"] == "human"
    assert {r["label"] for r in lines} <= set(demo.label_names)


def test_llm_lf_through_the_workbench(demo: Dataset) -> None:
    wb = Workbench(demo)
    fake = KeywordFakeProvider({"crashes": "bug", "wish": "feature"})
    column = wb.add_llm_lf("llm_rule", "Bug reports describe failures.", fake)
    assert column.kind == "llm"
    assert set(np.unique(column.train)) <= {ABSTAIN, 0, 1}
    with pytest.raises(ValueError, match="instruction"):
        wb.add_llm_lf("llm_empty", "  ", fake)
