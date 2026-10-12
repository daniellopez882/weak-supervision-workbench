"""An LLM as one more labelling function, behind a two-method provider interface.

``OpenAICompatibleProvider`` talks to any server that implements ``POST /chat/completions``
(OpenAI, Ollama, vLLM, LM Studio). ``KeywordFakeProvider`` answers deterministically from
keyword cues and is what the tests use; nothing in the benchmark calls a real model.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from wsw.lf import ABSTAIN

ENV_BASE_URL = "WSW_LLM_BASE_URL"
ENV_MODEL = "WSW_LLM_MODEL"
ENV_API_KEY = "WSW_LLM_API_KEY"
TEXT_MARKER = "Text:"
ABSTAIN_WORD = "ABSTAIN"

SYSTEM_PROMPT = (
    "You label short texts for a classification dataset. Reply with exactly one of the "
    f"allowed labels, or {ABSTAIN_WORD} if the rule below does not clearly apply. "
    "Reply with the label only."
)


class ChatProvider(Protocol):
    def complete(self, system: str, user: str) -> str: ...


@dataclass(frozen=True)
class OpenAICompatibleProvider:
    base_url: str
    model: str
    api_key: str | None = None
    timeout: float = 60.0
    max_tokens: int = 8

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> OpenAICompatibleProvider | None:
        """Configured from ``WSW_LLM_BASE_URL`` and ``WSW_LLM_MODEL``; ``None`` if unset."""
        env = os.environ if env is None else env
        base_url, model = env.get(ENV_BASE_URL), env.get(ENV_MODEL)
        if not base_url or not model:
            return None
        return cls(base_url.rstrip("/"), model, env.get(ENV_API_KEY) or None)

    def complete(self, system: str, user: str) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "temperature": 0,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return str(payload["choices"][0]["message"]["content"])


@dataclass
class KeywordFakeProvider:
    """Replies with the label of the first cue found in the text part of the prompt."""

    cues: Mapping[str, str]
    calls: int = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        text = user.rsplit(TEXT_MARKER, 1)[-1].lower()
        for cue, label in self.cues.items():
            if cue.lower() in text:
                return label
        return ABSTAIN_WORD


def build_prompt(instruction: str, label_names: Sequence[str], text: str) -> str:
    labels = ", ".join(label_names)
    return f"Rule: {instruction.strip()}\nAllowed labels: {labels}\n{TEXT_MARKER}\n{text}"


def parse_reply(reply: str, label_names: Sequence[str]) -> int:
    """Map a reply to a class id; anything that is not exactly one label name abstains."""
    words = {w.lower() for w in re.findall(r"[\w-]+", reply)}
    hits = [i for i, name in enumerate(label_names) if name.lower() in words]
    return hits[0] if len(hits) == 1 else ABSTAIN


@dataclass
class LLMLabelingFunction:
    name: str
    provider: ChatProvider
    label_names: tuple[str, ...]
    instruction: str
    # Texts repeat across splits and re-applications; one call per distinct text is enough.
    _cache: dict[str, int] = field(default_factory=dict, repr=False)

    def __call__(self, text: str) -> int:
        if text not in self._cache:
            reply = self.provider.complete(
                SYSTEM_PROMPT, build_prompt(self.instruction, self.label_names, text)
            )
            self._cache[text] = parse_reply(reply, self.label_names)
        return self._cache[text]
