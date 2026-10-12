from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from wsw.benchmark.machine import machine_info
from wsw.benchmark.report import learning_curve_svg, markdown_report
from wsw.benchmark.runner import SUITES, SuiteConfig, run_suite
from wsw.cli import main

TINY = SuiteConfig(
    name="smoke",
    datasets=("demo",),
    seeds=(0, 1),
    budgets=(10, 20),
    batch_size=10,
    al_pool_cap=300,
    tfidf={"ngram_range": (1, 1), "min_df": 1, "max_features": 5000},
    synthetic_n=2000,
    synthetic_m=5,
)


@pytest.fixture(scope="module")
def tiny_result(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict, Path]:
    out = tmp_path_factory.mktemp("bench") / "smoke-demo.json"
    run_suite(TINY, out, "wsw bench --suite smoke", use_snorkel=False, log=lambda _m: None)
    return json.loads(out.read_text(encoding="utf-8")), out


def test_results_carry_provenance(tiny_result: tuple[dict, Path]) -> None:
    res, _ = tiny_result
    for key in ("machine", "date", "git_sha", "command", "timestamp", "wall_seconds"):
        assert key in res
    assert res["command"] == "wsw bench --suite smoke"
    assert res["machine"]["python"].startswith("3.12")
    assert res["llm_labelling_function"] == "not run"
    assert res["snorkel_reference"].startswith("not run")


def test_results_cover_every_section(tiny_result: tuple[dict, Path]) -> None:
    res, _ = tiny_result
    lm = res["label_models"]["demo"]
    assert {"majority_vote", "dawid_skene", "metal", "flyingsquid"} <= set(lm)
    assert len(lm["metal"]["per_seed"]) == 2
    al = res["active_learning"]["demo"]
    assert set(al) == {
        "random",
        "conflict_entropy",
        "posterior_margin",
        "coreset",
        "random_gold_labels_only",
    }
    assert al["random"]["curve"]["n_labels"] == [0, 10, 20]
    assert al["random_gold_labels_only"]["curve"]["n_labels"] == [10, 20]
    assert al["random"]["nAULC_accuracy_minus_random"]["mean"] == 0
    assert set(res["synthetic_recovery"]) == {"k2", "k4"}
    assert 0 <= res["class_balance"]["demo"]["estimated_l1_error"] <= 2


def test_report_renders_tables_and_svg(tiny_result: tuple[dict, Path], capsys) -> None:
    res, path = tiny_result
    md = markdown_report(res)
    assert "| Majority vote |" in md
    assert "Snorkel" not in md.split("### End model")[0]  # snorkel not run, so no row
    root = ET.fromstring(learning_curve_svg(res))
    assert root.tag.endswith("svg")
    assert main(["report", str(path)]) == 0
    assert "normalised AULC" in capsys.readouterr().out


def test_full_suite_matches_the_documented_plan() -> None:
    full = SUITES["full"]
    assert full.datasets == ("youtube", "sms", "trec", "yelp", "agnews")
    assert len(full.seeds) == 5
    assert full.budgets == (25, 50, 100, 250, 500)


def test_machine_info_fields() -> None:
    info = machine_info()
    assert {"cpu", "logical_cpus", "ram_gb", "os", "python", "packages"} <= set(info)
    assert info["packages"]["numpy"] != "not installed"
