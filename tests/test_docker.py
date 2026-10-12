"""Static checks on the image definition (CI builds and smoke-runs it)."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _final_stage() -> list[str]:
    lines = [ln.strip() for ln in (ROOT / "Dockerfile").read_text(encoding="utf-8").splitlines()]
    starts = [i for i, ln in enumerate(lines) if ln.upper().startswith("FROM ")]
    return lines[starts[-1] :]


def test_final_stage_is_slim_and_non_root() -> None:
    stage = _final_stage()
    assert re.match(r"FROM python:3\.12-slim", stage[0])
    assert "USER 10001" in stage or "USER 10001:10001" in stage
    assert any(ln.startswith("HEALTHCHECK") for ln in stage)


def test_no_compiler_in_final_stage() -> None:
    text = "\n".join(_final_stage()).lower()
    for tool in ("gcc", "g++", "build-essential", "clang", "apt-get install"):
        assert tool not in text


def test_dockerignore_keeps_data_and_venv_out() -> None:
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").split()
    for entry in (".venv", ".git", "bench/results"):
        assert entry in ignored
