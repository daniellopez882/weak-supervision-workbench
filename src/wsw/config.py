"""Runtime locations. Nothing here may point at a specific machine."""

from __future__ import annotations

import os
from pathlib import Path

DATA_DIR_ENV = "WSW_DATA_DIR"


def data_dir() -> Path:
    """Root for downloaded data: ``$WSW_DATA_DIR``, else ``~/.cache/weak-supervision-workbench``."""
    override = os.environ.get(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "weak-supervision-workbench"
