"""Shared fixtures. Everything here exists so no test needs a live model.

No test in this suite requires Ollama running, a populated database, or a real
.env file. The .env part is only possible because app.config exposes a cached
get_settings() rather than a module-level singleton: a singleton would read the
environment at import time, before any fixture could set it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Same pattern as scripts/run_triage.py and eval/run_eval.py: pytest puts the
# test directory on sys.path, not the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import llm_client  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.llm_client import get_client  # noqa: E402

TRUNCATE = "__TRUNCATE__"  # sentinel: return this text with finish_reason=length


@pytest.fixture(autouse=True)
def settings_env(monkeypatch):
    """Provide config from the environment, so no .env file is consulted.

    Environment variables take priority over .env in pydantic-settings, so these
    win even when a real .env happens to exist. The caches are cleared on the
    way in and out so no value leaks between tests.
    """
    get_settings.cache_clear()
    get_client.cache_clear()
    # A dead port on purpose: if any test ever reaches the network instead of
    # the fake client, it fails loudly rather than quietly using a live Ollama.
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key-not-a-real-credential")
    monkeypatch.setenv("LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("LLM_MAX_TOKENS", "512")
    monkeypatch.setenv("CONFIDENCE_THRESHOLD", "0.8")
    yield
    get_settings.cache_clear()
    get_client.cache_clear()


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Skip the 1s + 2s retry backoff so the suite stays fast."""
    monkeypatch.setattr(llm_client.time, "sleep", lambda _seconds: None)


class FakeCompletions:
    """Plays back a scripted list of outcomes and counts how often it was called.

    Each script entry is either a string (returned as the message content), the
    TRUNCATE sentinel (returned with finish_reason="length"), or an exception
    instance (raised). The last entry repeats once the script runs out, which is
    what makes "always fails" and "always invalid" one-line scripts.
    """

    def __init__(self, script):
        self.script = script
        self.calls = 0
        self.messages_seen = []

    def create(self, **kwargs):
        self.messages_seen.append(kwargs.get("messages", []))
        item = self.script[min(self.calls, len(self.script) - 1)]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        finish = "length" if item is TRUNCATE else "stop"
        content = "" if item is TRUNCATE else item
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content), finish_reason=finish
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1300, completion_tokens=80),
        )


@pytest.fixture
def fake_llm(monkeypatch):
    """Substitute the OpenAI client, leaving the real call_llm to run.

    Mocking here rather than at app.triage.call_llm keeps the transport layer
    under test: the retry loop, the backoff, the exception classification and
    the truncation check are all real code in every test below.
    """

    def install(*script):
        completions = FakeCompletions(list(script))
        client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
        monkeypatch.setattr(llm_client, "get_client", lambda: client)
        return completions

    return install


def valid_json(**overrides) -> str:
    """A schema-valid response body, as bare JSON."""
    import json

    payload = {
        "category": "billing",
        "priority": "high",
        "summary": "Customer was charged twice and asks for the duplicate to be refunded.",
        "suggested_reply": "Thanks for flagging this - could you confirm the two invoice numbers?",
        "confidence": 0.9,
    }
    payload.update(overrides)
    return json.dumps(payload)


TICKET = {
    "id": "T-TEST",
    "subject": "Charged twice for June",
    "body": (
        "I was billed forty nine dollars twice for my June subscription and "
        "would like the duplicate charge refunded. The invoice references are "
        "INV-1 and INV-2."
    ),
}
