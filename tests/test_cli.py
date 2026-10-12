from __future__ import annotations

import pytest

from wsw.cli import build_parser, main


def test_lfs_command_prints_diagnostics(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["lfs", "demo"]) == 0
    out = capsys.readouterr().out
    assert "kw_crash" in out
    assert "coverage" in out


def test_parser_defaults() -> None:
    args = build_parser().parse_args(["bench"])
    assert args.suite == "full"
    args = build_parser().parse_args(["app"])
    assert (args.dataset, args.port, args.max_rows) == ("demo", 8080, 5000)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["fetch", "imdb"])


def test_app_refuses_a_dataset_that_is_not_downloaded(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("WSW_DATA_DIR", str(tmp_path))
    assert main(["app", "--dataset", "agnews"]) == 2
    assert "wsw fetch agnews" in capsys.readouterr().err
