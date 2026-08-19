"""The reliability layer, made to execute on demand.

In the 30-ticket run only one ticket fell back and two needed a repair, so most
of this code never ran. These tests run all of it without a live model.

The LLM is mocked at the OpenAI client, so the real call_llm executes in every
test here: its retry loop, backoff, exception classification and truncation
handling are under test rather than replaced.
"""

import json

import openai
import pytest

from app.llm_client import get_client
from app.schemas import Category, Priority
from app.triage import triage_ticket
from tests.conftest import TICKET, TRUNCATE, valid_json

# Copied verbatim from the Phase 0 run (T-018, run 1). Embedded rather than read
# from failures.txt because that file is gitignored, so a test reading it would
# pass here and fail on a clean clone.
#
# Note the shape: prose preamble, a bare ``` fence, and an extra `escalate` key
# the model was never asked for. All 8 Phase 0 responses looked like this and
# none was parseable by json.loads directly.
PHASE_0_RESPONSE = (
    "Here is the JSON representation of the support ticket:\n"
    "\n"
    "```\n"
    "{\n"
    '  "category": "bug",\n'
    '  "priority": "high",\n'
    '  "summary": "it doesn\'t work",\n'
    '  "suggested_reply": "Please try restarting your device or checking for any updates.",\n'
    '  "confidence": 0.8,\n'
    '  "escalate": true\n'
    "}\n"
    "```\n"
)

INVALID_RESPONSE = valid_json(confidence=100, category="not billing")


def test_prose_wrapped_fenced_json_is_extracted_and_validated(fake_llm):
    """1. The Phase 0 failure shape survives extraction and validates."""
    fake = fake_llm(PHASE_0_RESPONSE)
    record = triage_ticket(TICKET)

    assert fake.calls == 1
    assert record.fallback_used is False
    assert record.category is Category.BUG
    assert record.priority is Priority.HIGH
    assert record.model_confidence == 0.8
    # The extra `escalate: true` the model volunteered was dropped, not obeyed.
    assert record.summary == "it doesn't work"


def test_invalid_output_is_repaired_and_second_response_accepted(fake_llm):
    """2. A schema failure triggers one repair carrying the correction."""
    fake = fake_llm(INVALID_RESPONSE, valid_json())
    record = triage_ticket(TICKET)

    assert fake.calls == 2, "expected exactly one repair attempt"
    assert record.fallback_used is False
    assert record.category is Category.BILLING
    assert record.retries == 1

    # The repair conversation must carry the model's own bad output back, plus
    # the correction naming the rejected fields.
    repair_messages = fake.messages_seen[1]
    assert repair_messages[-2]["role"] == "assistant"
    assert repair_messages[-2]["content"] == INVALID_RESPONSE
    correction = repair_messages[-1]["content"]
    assert "confidence" in correction and "category" in correction
    assert "less than or equal to 1" in correction


def test_persistent_invalid_output_falls_back_without_inventing(fake_llm):
    """3. Budget exhausted -> fallback, with the summary from the BODY."""
    fake = fake_llm(INVALID_RESPONSE)
    record = triage_ticket(TICKET)

    assert fake.calls == 3
    assert record.fallback_used is True
    assert record.category is Category.OTHER
    assert record.priority is Priority.MEDIUM
    assert record.model_confidence == 0.0
    assert record.final_confidence == 0.0
    assert record.escalate is True
    assert record.escalation_reason

    # The summary is the ticket body truncated -- never the model's words.
    assert record.summary == TICKET["body"][:200]
    assert record.summary != json.loads(INVALID_RESPONSE)["summary"]
    # Nothing was drafted, and nothing was invented to fill the gap.
    assert record.suggested_reply == ""


def test_transport_failure_falls_back_rather_than_raising(fake_llm):
    """4. A dead connection produces a record, not an exception."""
    fake = fake_llm(openai.APIConnectionError(request=None))
    record = triage_ticket(TICKET)

    assert fake.calls == 3, "a retryable failure should use the whole budget"
    assert record.fallback_used is True
    assert record.escalate is True
    assert "transport" in record.escalation_reason
    assert record.retries == 2


@pytest.mark.parametrize(
    "script, label",
    [
        ([INVALID_RESPONSE], "always invalid schema"),
        ([openai.APIConnectionError(request=None)], "always failing transport"),
        # The decisive case. The first two scripts each exercise only ONE of the
        # two budgets, so a bug that gave each repair a fresh transport budget
        # would still show 3 calls and pass. Mixing them is what makes the
        # multiplication observable: one schema failure, then transport failures
        # inside the repair. Verified by mutation -- rewriting the repair call as
        # call_llm(messages) without max_attempts makes this case return 4.
        ([INVALID_RESPONSE, openai.APIConnectionError(request=None)],
         "invalid, then failing transport"),
    ],
)
def test_retry_budget_is_never_exceeded(fake_llm, script, label):
    """5. LLM_MAX_RETRIES=2 means exactly 3 model calls, by either route.

    This is the test that proves the transport budget and the repair budget add
    rather than multiply. If they nested, the invalid-schema path would make
    (2+1) x (2+1) = 9 calls.
    """
    fake = fake_llm(*script)
    record = triage_ticket(TICKET)

    assert fake.calls == 3, f"{label}: expected 3 calls, got {fake.calls}"
    assert record.fallback_used is True


def test_truncated_response_is_not_repaired(fake_llm):
    """6. finish_reason=length is deterministic, so retrying cannot help."""
    fake = fake_llm(TRUNCATE)
    record = triage_ticket(TICKET)

    assert fake.calls == 1, "a truncated response must not be repair-retried"
    assert record.fallback_used is True
    assert "truncated" in record.escalation_reason


def test_sdk_retries_are_disabled_on_the_real_client():
    """6b. The one regression no mocked test can see.

    Every other test replaces get_client, so the line that sets max_retries=0
    never runs in them. If it were deleted, the SDK would retry internally and
    multiply against our budget -- invisibly, since those attempts never reach
    our logs. This asserts on a real client; constructing one performs no I/O.
    """
    client = get_client()
    assert client.max_retries == 0
