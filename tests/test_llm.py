from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from wsw.lf import ABSTAIN, apply_lfs
from wsw.llm import (
    KeywordFakeProvider,
    LLMLabelingFunction,
    OpenAICompatibleProvider,
    build_prompt,
    parse_reply,
)

LABELS = ("bug", "feature", "praise")


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("bug", 0),
        ("  Feature\n", 1),
        ("Label: praise.", 2),
        ("ABSTAIN", ABSTAIN),
        ("bug or feature", ABSTAIN),  # two labels named: not an answer
        ("", ABSTAIN),
        ("debugging", ABSTAIN),  # a label inside another word does not count
    ],
)
def test_parse_reply(reply: str, expected: int) -> None:
    assert parse_reply(reply, LABELS) == expected


def test_llm_lf_with_fake_provider_caches_per_text() -> None:
    fake = KeywordFakeProvider({"crash": "bug", "please add": "feature"})
    lf = LLMLabelingFunction("llm", fake, LABELS, "Is this a bug report or a feature request?")
    texts = ["It crashes on start", "please add sync", "nice", "It crashes on start"]
    L = apply_lfs([lf], texts, k=3)
    assert L[:, 0].tolist() == [0, 1, ABSTAIN, 0]
    assert fake.calls == 3


def test_fake_provider_reads_only_the_text_part() -> None:
    # "bug" appears in the rule but not in the text, so the fake must abstain.
    prompt = build_prompt("Label bug reports as bug", LABELS, "great app")
    assert KeywordFakeProvider({"bug": "bug"}).complete("", prompt) == "ABSTAIN"


class _Handler(BaseHTTPRequestHandler):
    seen: list[dict] = []  # noqa: RUF012 (shared across handler instances on purpose)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Handler.seen.append({"path": self.path, "auth": self.headers.get("Authorization"), **body})
        payload = json.dumps({"choices": [{"message": {"content": "feature"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def local_server() -> Iterator[str]:
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        server.shutdown()
        server.server_close()


def test_openai_compatible_provider_request_shape(local_server: str) -> None:
    _Handler.seen.clear()
    provider = OpenAICompatibleProvider.from_env(
        {
            "WSW_LLM_BASE_URL": local_server + "/",
            "WSW_LLM_MODEL": "local-model",
            "WSW_LLM_API_KEY": "k",
        }
    )
    assert provider is not None
    lf = LLMLabelingFunction("llm", provider, LABELS, "Feature requests ask for something new.")
    assert lf("please add an export button") == 1
    request = _Handler.seen[0]
    assert request["path"] == "/v1/chat/completions"
    assert request["auth"] == "Bearer k"
    assert request["model"] == "local-model"
    assert request["temperature"] == 0
    assert request["messages"][1]["content"].endswith("please add an export button")


def test_provider_from_env_needs_url_and_model() -> None:
    assert OpenAICompatibleProvider.from_env({}) is None
    assert OpenAICompatibleProvider.from_env({"WSW_LLM_BASE_URL": "http://x"}) is None
