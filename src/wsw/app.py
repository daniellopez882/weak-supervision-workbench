"""NiceGUI front end: write LFs, inspect their conflicts, label the queue, train and export.

Each browser tab gets its own ``Workbench`` built by the factory passed to ``make_root``;
datasets are loaded once per process and shared read-only.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import numpy as np
from nicegui import app, run, ui

from wsw.active.strategies import STRATEGIES
from wsw.data import cap_train_rows, load_dataset
from wsw.data.dataset import Dataset
from wsw.labelmodels import LABEL_MODELS
from wsw.llm import ChatProvider, OpenAICompatibleProvider
from wsw.workbench import Workbench

TITLE = "Weak supervision workbench"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


class WorkbenchPage:
    def __init__(self, wb: Workbench, provider: ChatProvider | None) -> None:
        self.wb = wb
        self.provider = provider
        self.current = wb.next_item()

    # Layout ---------------------------------------------------------------------------------

    def build(self) -> None:
        ui.page_title(TITLE)
        ui.query("body").style("background-color: #f6f6f4")
        ds = self.wb.dataset
        with ui.header().classes("items-center justify-between"):
            ui.label(TITLE).classes("text-lg font-semibold")
            ui.label(
                f"{ds.name}: {len(ds.train)} training rows, classes {', '.join(ds.label_names)}"
            ).mark("dataset-info")
        with ui.row().classes("items-end gap-6 q-pa-sm"):
            ui.select(
                sorted(LABEL_MODELS),
                value=self.wb.label_model_name,
                label="Label model",
                on_change=lambda e: self._set_label_model(str(e.value)),
            ).classes("w-48").mark("label-model")
            ui.select(
                sorted(STRATEGIES),
                value=self.wb.strategy_name,
                label="Queue strategy",
                on_change=lambda e: self._set_strategy(str(e.value)),
            ).classes("w-48").mark("strategy")
            self.summary_view()
        with ui.tabs() as tabs:
            write = ui.tab("Write LFs")
            conflicts = ui.tab("Conflicts")
            queue = ui.tab("Label queue")
            train = ui.tab("Train and export")
        with ui.tab_panels(tabs, value=write).classes("w-full"):
            with ui.tab_panel(write):
                self.lf_form()
                self.lf_table()
            with ui.tab_panel(conflicts):
                self.conflict_view()
            with ui.tab_panel(queue):
                self.queue_view()
            with ui.tab_panel(train):
                self.train_view()

    @ui.refreshable_method
    def summary_view(self) -> None:
        s = self.wb.summary()
        ui.label(
            f"{s.n_lfs} LFs | coverage {_pct(s.coverage)} | "
            f"{s.mean_votes_per_covered_row:.2f} votes per covered row | "
            f"rows with conflicting votes {_pct(s.conflict_rate)} | "
            f"{len(self.wb.gold)} human labels"
        ).mark("summary")

    def refresh_all(self) -> None:
        self.summary_view.refresh()
        self.lf_table.refresh()
        self.conflict_view.refresh()
        self.queue_view.refresh()

    # Write LFs --------------------------------------------------------------------------------

    def lf_form(self) -> None:
        kinds = ["keyword", "regex"] + (["llm"] if self.provider is not None else [])
        with ui.card().classes("w-full"):
            ui.label("New labelling function").classes("font-semibold")
            with ui.row().classes("items-end gap-4 w-full"):
                kind = ui.toggle(kinds, value="keyword").mark("lf-kind")
                name = ui.input("Name", placeholder="kw_refund").mark("lf-name")
                body = (
                    ui.input("Keywords, pattern or instruction", placeholder="refund, money back")
                    .classes("grow")
                    .mark("lf-body")
                )
                label = ui.select(
                    dict(enumerate(self.wb.label_names)), value=0, label="Votes for"
                ).mark("lf-label")
                ui.button("Add LF", on_click=lambda: self._add_lf(kind, name, body, label)).mark(
                    "lf-add"
                )
            if self.provider is None:
                ui.label(
                    "LLM labelling functions appear here when WSW_LLM_BASE_URL and "
                    "WSW_LLM_MODEL point at an OpenAI-compatible server."
                ).classes("text-xs text-gray-500")

    async def _add_lf(
        self, kind: ui.toggle, name_in: ui.input, body_in: ui.input, label_in: ui.select
    ) -> None:
        name, body, label = name_in.value or "", body_in.value or "", int(label_in.value)
        try:
            if kind.value == "keyword":
                column = self.wb.add_keyword_lf(name, body, label)
            elif kind.value == "regex":
                column = self.wb.add_regex_lf(name, body, label)
            else:
                assert self.provider is not None
                ui.notify(f"Asking the model about {len(self.wb.texts)} rows...")
                provider = self.provider
                column = await run.io_bound(self.wb.add_llm_lf, name, body, provider)
        except ValueError as exc:
            ui.notify(str(exc), type="warning")
            return
        name_in.value, body_in.value = "", ""
        stats = next(s for s in self.wb.stats() if s.name == column.name)
        ui.notify(
            f"{column.name}: coverage {_pct(stats.coverage)}, conflicts {_pct(stats.conflict)}",
            type="positive",
        )
        self.current = self.wb.next_item()
        self.refresh_all()

    @ui.refreshable_method
    def lf_table(self) -> None:
        names = self.wb.label_names
        estimated = self.wb.estimated_accuracy()
        rows = []
        for j, (col, s) in enumerate(zip(self.wb.columns, self.wb.stats(), strict=True)):
            est = None if estimated is None else estimated[j]
            rows.append(
                {
                    "name": s.name,
                    "kind": col.kind,
                    "votes": ", ".join(names[c] for c in s.polarity) or "-",
                    "coverage": _pct(s.coverage),
                    "overlap": _pct(s.overlap),
                    "conflict": _pct(s.conflict),
                    "human_acc": "-"
                    if s.empirical_accuracy is None
                    else f"{_pct(s.empirical_accuracy)} of {s.n_labelled_votes}",
                    "model_acc": "-" if est is None or est != est else _pct(float(est)),
                    "rule": col.description,
                }
            )
        columns = [
            {"name": key, "label": label, "field": key, "align": "left"}
            for key, label in [
                ("name", "LF"),
                ("kind", "Kind"),
                ("votes", "Votes for"),
                ("coverage", "Coverage"),
                ("overlap", "Overlap"),
                ("conflict", "Conflict"),
                ("human_acc", "Accuracy on human labels"),
                ("model_acc", "Accuracy estimated by label model"),
                ("rule", "Rule"),
            ]
        ]
        ui.table(columns=columns, rows=rows, row_key="name", pagination=15).classes("w-full").mark(
            "lf-table"
        )
        if self.wb.columns:
            with ui.row().classes("items-end gap-4"):
                target = ui.select(self.wb.lf_names, label="LF to remove").classes("w-64")
                target.mark("lf-remove-select")
                ui.button(
                    "Remove", on_click=lambda: self._remove_lf(target.value), color="negative"
                ).mark("lf-remove")

    def _remove_lf(self, name: str | None) -> None:
        if not name:
            ui.notify("Pick an LF first", type="warning")
            return
        self.wb.remove_lf(name)
        self.current = self.wb.next_item()
        self.refresh_all()

    # Conflicts --------------------------------------------------------------------------------

    @ui.refreshable_method
    def conflict_view(self) -> None:
        names = self.wb.lf_names
        if len(names) < 2:
            ui.label("Add at least two LFs to see how they overlap and conflict.")
            return
        pairs = self.wb.pairs()
        with ui.row().classes("w-full gap-4 no-wrap"):
            for title, matrix in (("Overlap", pairs.overlap), ("Conflict", pairs.conflict)):
                # The overlap diagonal is each LF's coverage (shown in the table); leaving it
                # out keeps the colour scale for the pairs.
                off_diagonal = matrix[~np.eye(len(names), dtype=bool)]
                data = [
                    [i, j, "-" if i == j else round(float(matrix[j, i]), 4)]
                    for i in range(len(names))
                    for j in range(len(names))
                ]
                ui.echart(
                    {
                        "title": {"text": f"{title} between LF pairs (fraction of rows)"},
                        "tooltip": {"position": "top"},
                        "grid": {"left": 120, "right": 20, "bottom": 130, "top": 40},
                        "xAxis": {"type": "category", "data": names, "axisLabel": {"rotate": 60}},
                        "yAxis": {"type": "category", "data": names, "inverse": True},
                        "visualMap": {
                            "min": 0,
                            "max": max(float(off_diagonal.max()), 1e-6),
                            "calculable": True,
                            "precision": 2,
                            "orient": "horizontal",
                            "left": "center",
                            "bottom": 0,
                            "inRange": {"color": ["#eef4fc", "#2a78d6", "#0d366b"]},
                        },
                        "series": [{"type": "heatmap", "data": data}],
                    }
                ).classes("w-1/2 h-[28rem]").mark(f"heatmap-{title.lower()}")
        ui.label("Rows where the LFs disagree most").classes("font-semibold q-mt-md")
        for item in self.wb.most_conflicted(10):
            with ui.card().classes("w-full"):
                ui.label(item.text)
                ui.label(
                    "; ".join(f"{lf} -> {lab}" for lf, lab in item.votes)
                    + f" | conflict entropy {item.conflict:.2f}"
                ).classes("text-xs text-gray-600")

    # Queue ------------------------------------------------------------------------------------

    @ui.refreshable_method
    def queue_view(self) -> None:
        ui.label(
            f"{len(self.wb.gold)} labelled, {len(self.wb.skipped)} skipped, "
            f"strategy: {self.wb.strategy_name}"
        ).mark("queue-progress")
        item = self.current
        if item is None:
            ui.label("The queue is empty: every row is labelled or skipped.")
            return
        with ui.card().classes("w-full"):
            ui.label(item.text).classes("text-base").mark("queue-text")
            votes = "; ".join(f"{lf} -> {lab}" for lf, lab in item.votes) or "no LF votes"
            ui.label(f"LF votes: {votes}").classes("text-xs text-gray-600")
            posterior = ", ".join(
                f"{name} {p:.2f}"
                for name, p in zip(self.wb.label_names, item.posterior, strict=True)
            )
            ui.label(
                f"Label model: {posterior} | conflict entropy {item.conflict:.2f} | "
                f"margin {item.margin:.2f}"
            ).classes("text-xs text-gray-600")
            with ui.row():
                for c, name in enumerate(self.wb.label_names):
                    ui.button(name, on_click=functools.partial(self._answer, item.index, c)).mark(
                        f"answer-{name}"
                    )
                ui.button("Skip", on_click=lambda: self._skip(item.index), color="grey").mark(
                    "skip"
                )

    def _answer(self, index: int, label: int) -> None:
        self.wb.submit(index, label)
        self.current = self.wb.next_item()
        self.refresh_all()

    def _skip(self, index: int) -> None:
        self.wb.skip(index)
        self.current = self.wb.next_item()
        self.queue_view.refresh()

    def _set_label_model(self, name: str) -> None:
        self.wb.set_label_model(name)
        self.current = self.wb.next_item()
        self.refresh_all()

    def _set_strategy(self, name: str) -> None:
        self.wb.set_strategy(name)
        self.current = self.wb.next_item()
        self.queue_view.refresh()

    # Train and export -------------------------------------------------------------------------

    def train_view(self) -> None:
        ui.label(
            "Trains the TF-IDF + logistic-regression end model on the label model's soft labels "
            "(human labels replace them where present) and scores both models on the test split."
        ).classes("text-sm")
        results = ui.column().mark("train-results")

        async def train() -> None:
            try:
                ev = await run.io_bound(self.wb.evaluate)
            except ValueError as exc:
                ui.notify(str(exc), type="warning")
                return
            results.clear()
            with results:
                ui.label(
                    f"Label model ({self.wb.label_model_name}): accuracy "
                    f"{_pct(ev.label_model_accuracy)}, F1 {_pct(ev.label_model_f1)}"
                )
                ui.label(
                    f"End model: accuracy {_pct(ev.end_model_accuracy)}, "
                    f"F1 {_pct(ev.end_model_f1)} ({ev.train_rows} training rows, "
                    f"{ev.human_labels} human labels, {ev.seconds:.1f} s)"
                ).mark("end-model-result")

        def export() -> None:
            ui.download.content(
                self.wb.export_jsonl(), f"{self.wb.dataset.name}-labels.jsonl", "application/jsonl"
            )

        with ui.row():
            ui.button("Train and evaluate", on_click=train).mark("train")
            ui.button("Download labels (JSONL)", on_click=export).mark("export")


def healthz() -> dict[str, str]:
    return {"status": "ok"}


def make_root(
    factory: Callable[[], Workbench], provider: ChatProvider | None = None
) -> Callable[[], None]:
    def root() -> None:
        WorkbenchPage(factory(), provider).build()

    return root


@functools.cache
def _load(name: str, max_rows: int | None) -> Dataset:
    return cap_train_rows(load_dataset(name), max_rows)


def serve(
    dataset: str = "demo",
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    max_rows: int | None = 5000,
    label_model: str = "dawid_skene",
    strategy: str = "conflict_entropy",
) -> None:
    _load(dataset, max_rows)  # fail fast on a missing dataset, before the server starts

    def factory() -> Workbench:
        return Workbench(_load(dataset, max_rows), label_model=label_model, strategy=strategy)

    app.add_api_route("/healthz", healthz, methods=["GET"])
    ui.run(
        make_root(factory, OpenAICompatibleProvider.from_env()),
        host=host,
        port=port,
        title=TITLE,
        reload=False,
        show=False,
    )
