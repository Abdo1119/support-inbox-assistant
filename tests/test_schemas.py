"""The validation station. No mocking needed -- this is pure Pydantic.

Three of these four values are real Phase 0 output, not invented cases.
"""

import pytest
from pydantic import ValidationError

from app.schemas import Category, LLMTriageOutput, Priority, normalize_llm_payload

BASE = {
    "category": "billing",
    "priority": "high",
    "summary": "Customer was charged twice and asks for a refund.",
    "suggested_reply": "Thanks for flagging this - could you confirm the invoice numbers?",
    "confidence": 0.8,
}


def error_types(exc: ValidationError) -> dict[str, str]:
    return {".".join(str(p) for p in e["loc"]): e["type"] for e in exc.errors()}


def test_confidence_of_100_is_rejected():
    """7. The injection ticket returned exactly this in Phase 0."""
    with pytest.raises(ValidationError) as exc:
        LLMTriageOutput(**{**BASE, "confidence": 100})
    assert error_types(exc.value)["confidence"] == "less_than_equal"


def test_null_suggested_reply_is_rejected():
    """8. T-030 returned exactly this in Phase 0."""
    with pytest.raises(ValidationError) as exc:
        LLMTriageOutput(**{**BASE, "suggested_reply": None})
    assert error_types(exc.value)["suggested_reply"] == "string_type"


def test_not_billing_is_rejected_and_never_coerced_to_billing():
    """9. The line between normalizing form and guessing meaning.

    "not billing" contains "billing". Any substring match, any fuzzy lookup, any
    "close enough" mapping added later turns a visible rejection the system can
    repair into a confident wrong answer it cannot. This test fails the moment
    that happens.
    """
    for value in ("not billing", "billing issue", "BILLING!", "bill"):
        payload = normalize_llm_payload({**BASE, "category": value})
        with pytest.raises(ValidationError) as exc:
            LLMTriageOutput(**payload)
        assert error_types(exc.value)["category"] == "enum", value

    # And explicitly: none of them became billing by another route.
    assert normalize_llm_payload({"category": "not billing"})["category"] != Category.BILLING.value


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("  Billing  ", Category.BILLING),
        ("feature_request", Category.FEATURE_REQUEST),  # the labels.json spelling
        ("FEATURE_REQUEST", Category.FEATURE_REQUEST),
    ],
)
def test_form_is_normalized_without_changing_meaning(raw, expected):
    """10. Whitespace, case and underscores are form. They are safe to fix."""
    output = LLMTriageOutput(**normalize_llm_payload({**BASE, "category": raw}))
    assert output.category is expected
    # Priority goes through the same normalizer.
    assert LLMTriageOutput(
        **normalize_llm_payload({**BASE, "priority": " URGENT "})
    ).priority is Priority.URGENT
