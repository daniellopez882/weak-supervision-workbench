from __future__ import annotations

import contextlib
import hashlib
import io
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import requires_wrench
from wsw.data import cap_train_rows, downloaded_datasets, load_dataset, wrench
from wsw.data.synthetic import sample_label_matrix
from wsw.lf import ABSTAIN


def _write_fake_wrench(root: Path, name: str) -> dict[str, bytes]:
    files: dict[str, bytes] = {"label.json": json.dumps({"0": "ham", "1": "spam"}).encode()}
    for split, n in (("train", 6), ("valid", 2), ("test", 3)):
        rows = {
            str(i): {
                "label": i % 2,
                "weak_labels": [i % 2, -1, 1],
                "data": {"text": f"{split} {i}"},
            }
            for i in range(n)
        }
        files[f"{split}.json"] = json.dumps(rows).encode()
    folder = root / "wrench" / name
    folder.mkdir(parents=True)
    for filename, payload in files.items():
        (folder / filename).write_bytes(payload)
    return files


def test_wrench_loader_reads_the_json_layout(tmp_path: Path) -> None:
    _write_fake_wrench(tmp_path, "youtube")
    ds = wrench.load("youtube", tmp_path)
    assert ds.label_names == ["ham", "spam"]
    assert (len(ds.train), len(ds.valid), len(ds.test)) == (6, 2, 3)
    assert ds.lf_names == ["youtube_lf00", "youtube_lf01", "youtube_lf02"]
    np.testing.assert_array_equal(ds.train.L[1], [1, ABSTAIN, 1])
    assert ds.train.texts[5] == "train 5"
    assert "youtube" not in downloaded_datasets(tmp_path / "elsewhere")
    assert "youtube" in downloaded_datasets(tmp_path)


def test_missing_dataset_has_a_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="wsw fetch sms"):
        wrench.load("sms", tmp_path)
    with pytest.raises(ValueError, match="unknown dataset"):
        load_dataset("imdb")


@contextlib.contextmanager
def _serve(payloads: dict[str, bytes], url: str, timeout: float) -> Iterator[io.BytesIO]:
    yield io.BytesIO(payloads[url.rsplit("/", 1)[-1]])


def test_fetch_verifies_sha256(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "source"
    payloads = _write_fake_wrench(source, "sms")
    digests = {f"sms/{k}": hashlib.sha256(v).hexdigest() for k, v in payloads.items()}
    monkeypatch.setattr(wrench, "committed_checksums", lambda: digests)
    monkeypatch.setattr(
        wrench.urllib.request, "urlopen", lambda url, timeout: _serve(payloads, url, timeout)
    )
    target = tmp_path / "data"
    observed = wrench.fetch("sms", target, log=lambda _msg: None)
    assert observed == digests
    assert wrench.is_available("sms", target)
    # A second fetch finds the verified files and does not download again.
    monkeypatch.setattr(wrench.urllib.request, "urlopen", lambda url, timeout: 1 / 0)
    assert wrench.fetch("sms", target, log=lambda _msg: None) == digests

    tampered = dict(digests, **{"sms/test.json": "0" * 64})
    monkeypatch.setattr(wrench, "committed_checksums", lambda: tampered)
    monkeypatch.setattr(
        wrench.urllib.request, "urlopen", lambda url, timeout: _serve(payloads, url, timeout)
    )
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        wrench.fetch("sms", tmp_path / "other", log=lambda _msg: None)
    assert not list((tmp_path / "other" / "wrench" / "sms").glob("*.part"))


def test_committed_checksums_cover_every_file() -> None:
    expected = {f"{n}/{f}" for n in wrench.WRENCH_DATASETS for f in wrench.FILES}
    assert set(wrench.committed_checksums()) == expected


def test_demo_dataset_is_deterministic() -> None:
    a, b = load_dataset("demo"), load_dataset("demo")
    assert a.train.texts == b.train.texts
    np.testing.assert_array_equal(a.train.L, b.train.L)
    assert a.k == 3
    assert a.m == 8


def test_cap_train_rows() -> None:
    ds = load_dataset("demo")
    small = cap_train_rows(ds, 100, seed=2)
    assert len(small.train) == 100
    assert len(small.test) == len(ds.test)
    assert cap_train_rows(ds, None) is ds
    assert "100 of" in small.notes["train_rows"]


def test_synthetic_matrix_follows_its_parameters() -> None:
    data = sample_label_matrix(
        50_000, np.array([0.8, 0.6]), np.array([0.5, 0.2]), np.array([0.6, 0.4]), seed=0
    )
    fired = data.L != ABSTAIN
    np.testing.assert_allclose(fired.mean(axis=0), [0.5, 0.2], atol=0.01)
    right = (data.L == data.y[:, None]) & fired
    np.testing.assert_allclose(right.sum(axis=0) / fired.sum(axis=0), [0.8, 0.6], atol=0.01)
    with pytest.raises(ValueError):
        sample_label_matrix(10, np.array([0.8]), np.array([0.5, 0.5]), np.array([0.5, 0.5]), 0)


@pytest.mark.integration
@requires_wrench
def test_real_wrench_datasets_load() -> None:
    shapes = {
        "youtube": (2, 10),
        "sms": (2, 73),
        "trec": (6, 68),
        "yelp": (2, 8),
        "agnews": (4, 9),
    }
    for name, (k, m) in shapes.items():
        ds = load_dataset(name)
        assert (ds.k, ds.m) == (k, m)
        assert len(ds.test) > 0
