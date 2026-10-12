"""UI tests with NiceGUI's simulated user (no browser, no server)."""

from __future__ import annotations

from nicegui import ui
from nicegui.testing.user_simulation import user_simulation

from wsw.app import healthz, make_root
from wsw.data import load_dataset
from wsw.llm import KeywordFakeProvider
from wsw.workbench import Workbench

DEMO = load_dataset("demo")


def _factory() -> Workbench:
    return Workbench(DEMO, label_model="dawid_skene", strategy="conflict_entropy")


def _table_rows(user) -> list[dict]:
    table = next(iter(user.find(marker="lf-table").elements))
    assert isinstance(table, ui.table)
    return table.rows


async def test_write_an_lf_and_see_it_in_the_table() -> None:
    async with user_simulation(root=make_root(_factory)) as user:
        await user.open("/")
        await user.should_see("Weak supervision workbench")
        await user.should_see("8 LFs")
        user.find(marker="lf-name").type("kw_sync")
        user.find(marker="lf-body").type("sync, export")
        user.find(marker="lf-add").click()
        await user.should_see("9 LFs")
        name_input = next(iter(user.find(marker="lf-name").elements))
        assert isinstance(name_input, ui.input)
        assert name_input.value == ""  # cleared for the next LF
        rows = _table_rows(user)
        added = next(r for r in rows if r["name"] == "kw_sync")
        assert added["kind"] == "keyword"
        assert added["votes"] == "bug"


async def test_bad_lf_input_is_reported_not_raised() -> None:
    async with user_simulation(root=make_root(_factory)) as user:
        await user.open("/")
        user.find(marker="lf-name").type("kw_empty")
        user.find(marker="lf-add").click()
        await user.should_see("needs at least one keyword")
        await user.should_see("8 LFs")


async def test_label_the_queue() -> None:
    async with user_simulation(root=make_root(_factory)) as user:
        await user.open("/")
        await user.should_see("0 labelled, 0 skipped")
        user.find(marker="answer-bug").click()
        await user.should_see("1 labelled, 0 skipped")
        user.find(marker="skip").click()
        await user.should_see("1 labelled, 1 skipped")
        await user.should_see("1 human labels")


async def test_conflicts_tab_has_heatmaps() -> None:
    async with user_simulation(root=make_root(_factory)) as user:
        await user.open("/")
        await user.should_see(marker="heatmap-overlap")
        await user.should_see(marker="heatmap-conflict")
        await user.should_see("Rows where the LFs disagree most")


async def test_train_and_evaluate() -> None:
    async with user_simulation(root=make_root(_factory)) as user:
        await user.open("/")
        user.find(marker="train").click()
        await user.should_see("End model: accuracy", retries=50)


async def test_llm_option_only_with_a_provider() -> None:
    async with user_simulation(root=make_root(_factory)) as user:
        await user.open("/")
        await user.should_see("LLM labelling functions appear here")
    fake = KeywordFakeProvider({"crashes": "bug"})
    async with user_simulation(root=make_root(_factory, fake)) as user:
        await user.open("/")
        await user.should_not_see("LLM labelling functions appear here")


def test_healthz() -> None:
    assert healthz() == {"status": "ok"}
