"""WRENCH text-classification subsets: fetch (sha256-verified) and load.

The WRENCH authors moved the data from Google Drive to the Hugging Face dataset
``jieyuz2/WRENCH``; we download from there at a pinned revision so the bytes are stable.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
from collections.abc import Callable
from importlib import resources
from pathlib import Path

import numpy as np

from wsw.config import data_dir
from wsw.data.dataset import Dataset, Split
from wsw.lf import ABSTAIN, LABEL_DTYPE

HF_REPO = "jieyuz2/WRENCH"
HF_REVISION = "d1185f6cfba8d5ace2a39229799cb3f1c136bf66"
BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_REVISION}/classification"
WRENCH_DATASETS = ("youtube", "sms", "trec", "yelp", "agnews")
FILES = ("label.json", "train.json", "valid.json", "test.json")
CHUNK = 1 << 20


def wrench_dir(root: Path | None = None) -> Path:
    return (root or data_dir()) / "wrench"


def committed_checksums() -> dict[str, str]:
    text = resources.files("wsw.data").joinpath("wrench_sha256.json").read_text("utf-8")
    return dict(json.loads(text))


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def is_available(name: str, root: Path | None = None) -> bool:
    folder = wrench_dir(root) / name
    return all((folder / f).is_file() for f in FILES)


def fetch(
    name: str,
    root: Path | None = None,
    *,
    record: bool = False,
    log: Callable[[str], None] = print,
) -> dict[str, str]:
    """Download one dataset and verify every file against the committed sha256 values.

    ``record=True`` accepts files that have no committed checksum yet and returns the
    observed digests so they can be committed; a mismatch with a committed value always fails.
    """
    if name not in WRENCH_DATASETS:
        raise ValueError(f"unknown WRENCH dataset {name!r}; choose from {WRENCH_DATASETS}")
    expected = committed_checksums()
    folder = wrench_dir(root) / name
    folder.mkdir(parents=True, exist_ok=True)
    observed: dict[str, str] = {}
    for filename in FILES:
        key = f"{name}/{filename}"
        target = folder / filename
        if target.is_file() and expected.get(key) == sha256_of(target):
            log(f"ok      {key} (cached)")
            observed[key] = expected[key]
            continue
        url = f"{BASE_URL}/{name}/{filename}"
        log(f"fetch   {url}")
        with tempfile.NamedTemporaryFile(dir=folder, delete=False, suffix=".part") as tmp:
            tmp_path = Path(tmp.name)
            with urllib.request.urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, tmp, CHUNK)
        digest = sha256_of(tmp_path)
        want = expected.get(key)
        if want is None and not record:
            tmp_path.unlink()
            raise RuntimeError(f"no committed sha256 for {key}; rerun with --record to add one")
        if want is not None and want != digest:
            tmp_path.unlink()
            raise RuntimeError(f"sha256 mismatch for {key}: expected {want}, got {digest}")
        tmp_path.replace(target)
        observed[key] = digest
        log(f"ok      {key} sha256={digest}")
    return observed


def _read_split(path: Path) -> Split:
    raw: dict[str, dict] = json.loads(path.read_text("utf-8"))
    rows = [raw[key] for key in sorted(raw, key=int)]
    texts = [str(r["data"]["text"]) for r in rows]
    L = np.asarray([r["weak_labels"] for r in rows], dtype=LABEL_DTYPE)
    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    return Split(texts, L.reshape(len(rows), -1), y)


def load(name: str, root: Path | None = None) -> Dataset:
    folder = wrench_dir(root) / name
    if not is_available(name, root):
        raise FileNotFoundError(f"{folder} is incomplete; run `wsw fetch {name}` first")
    labels: dict[str, str] = json.loads((folder / "label.json").read_text("utf-8"))
    label_names = [labels[str(i)] for i in range(len(labels))]
    train, valid, test = (_read_split(folder / f"{s}.json") for s in ("train", "valid", "test"))
    lf_names = [f"{name}_lf{j:02d}" for j in range(train.L.shape[1])]
    if (train.L < ABSTAIN).any():
        raise ValueError(f"{name}: unexpected negative weak label")
    return Dataset(
        name=name,
        label_names=label_names,
        lf_names=lf_names,
        train=train,
        valid=valid,
        test=test,
        notes={"source": f"WRENCH via huggingface.co/datasets/{HF_REPO}@{HF_REVISION[:12]}"},
    )
