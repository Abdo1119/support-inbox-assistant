"""Schemas for triage: what the model may claim, and what the system stores.

The two models here encode a trust boundary.

LLMTriageOutput is the contract for untrusted model output. Its field set is the
attack surface: five fields, and `escalate` is deliberately not one of them. A
field that does not exist cannot be injected -- the Phase 0 injection ticket got
`escalate: false`, exactly what the attacker asked for.

TriageRecord is what the system stores, including the fields only the system may
compute. The two have opposite strictness on purpose: the model may never claim
an empty summary, and the system must always be able to store one, because the
fallback path legitimately has nothing to say.

Every constraint below traces to something observed in Phase 0, not to a guess.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)


class Category(str, Enum):
    """Allowed ticket categories.

    Values use the brief's spelling -- `feature request` with a space -- rather
    than labels.json's `feature_request`. eval/results.json is the graded
    artefact and is compared against the brief, so the canonical form matches
    the output form and the final write needs no conversion. A conversion that
    does not exist cannot be the one that is forgotten.

    Conversion INTO this form happens in two inbound places only:
    normalize_llm_payload() below, and label loading in the eval. Nothing
    outbound converts.
    """

    BILLING = "billing"
    BUG = "bug"
    FEATURE_REQUEST = "feature request"
    ACCOUNT = "account"
    SECURITY = "security"
    OTHER = "other"


class Priority(str, Enum):
    """Allowed priorities, ordered low to urgent."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


# strip_whitespace runs before the length check, so "   " collapses to "" and is
# rejected rather than passing as three characters.
#
# Both floors sit BELOW the shortest output actually observed in Phase 0
# (summary 11 chars, reply 62). A floor above the real minimum would reject
# genuine output, burn a repair retry, and end at the fallback -- which replaces
# the summary with a truncated ticket body. Trading a mediocre summary for a
# truncated body makes the record worse, so these floors reject absence dressed
# as presence ("n/a", "none", "TBD", "-"), not mediocrity.
#
# What they do NOT catch: 7 of 8 Phase 0 runs returned the ticket subject
# verbatim as the summary, all of them 11 characters or more. That is a content
# failure, invisible to a length constraint, and belongs to the confidence
# signals rather than here.
Summary = Annotated[str, StringConstraints(strip_whitespace=True, min_length=5)]
SuggestedReply = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=20)
]


class LLMTriageOutput(BaseModel):
    """Exactly what the model is permitted to claim.

    There is no `escalate` field. Escalation is a system decision, computed from
    signals the model cannot influence.
    """

    # Unknown keys are dropped rather than rejected. All 8 Phase 0 responses
    # emitted `escalate` unprompted, so forbidding extras would fail every
    # ticket, exhaust the retry budget, and land everything in the fallback --
    # which sets escalate=true anyway. Dropping is not a silent failure:
    # unexpected_fields() below reports what was discarded, so the caller can
    # record it and the README can say how often it happened.
    model_config = ConfigDict(extra="ignore")

    category: Category
    priority: Priority
    summary: Summary
    suggested_reply: SuggestedReply
    # One Phase 0 response returned 100. Bounding it here makes that a
    # validation error carrying a repairable message, not a stored number.
    confidence: float = Field(ge=0.0, le=1.0)


class TriageRecord(BaseModel):
    """One triaged ticket as the system stores it.

    The content fields are plain `str` with no minimum length: the fallback path
    stores "" when it has nothing to say, and T-030's truncated body IS the
    empty string. The asymmetry with LLMTriageOutput is the point. A placeholder
    sentence would be text the system authored implying knowledge the ticket
    does not contain; escalation_reason carries the explanation instead.
    """

    ticket_id: str = Field(min_length=1)

    category: Category
    priority: Priority
    summary: str
    suggested_reply: str

    # Kept, not discarded. It is not a measurement -- it is a generated token
    # shaped like a number -- but the README's calibration histogram needs the
    # model's raw claim surviving next to the confidence the system computed.
    model_confidence: float = Field(ge=0.0, le=1.0)
    final_confidence: float = Field(ge=0.0, le=1.0)

    escalate: bool
    escalation_reason: str | None = None
    fallback_used: bool
    retries: int = Field(ge=0)

    @model_validator(mode="after")
    def _reason_matches_escalation(self) -> TriageRecord:
        """Enforce: escalate is true if and only if a reason is recorded.

        An escalated ticket with no reason is the failure that "a reason string,
        not just a boolean" exists to prevent. A reason on a ticket that was not
        escalated means the two fields disagree about what happened. Neither
        state is representable.
        """
        if self.escalate and not self.escalation_reason:
            raise ValueError("escalate=True requires a non-empty escalation_reason")
        if not self.escalate and self.escalation_reason:
            raise ValueError("escalation_reason is set but escalate=False")
        return self


# Only the two enum fields are normalized. The free-text fields are stored as
# the model wrote them.
_NORMALIZED_FIELDS = ("category", "priority")


def normalize_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize the FORM of the enum fields, before validation runs.

    Strips surrounding whitespace, lowercases, and maps underscores to spaces,
    so "  Feature_Request " becomes "feature request" and validates. This is
    what reconciles labels.json's `feature_request` spelling with the canonical
    `feature request`.

    It never normalizes MEANING. There is no substring matching and no mapping
    of near-misses: "not billing" and "billing issue" pass through unchanged and
    fail validation. That is deliberate. Coercing "not billing" to `billing`
    would convert a rejection the system can see and repair into a confident
    wrong answer it cannot -- and a wrong category is the most expensive error
    here, because a human triages from it.

    Non-string values are left alone so that validation reports the type error
    rather than this function hiding it. Returns a copy; the caller keeps the
    raw payload intact for logging.
    """
    normalized = dict(payload)
    for field in _NORMALIZED_FIELDS:
        value = normalized.get(field)
        if isinstance(value, str):
            normalized[field] = value.strip().lower().replace("_", " ")
    return normalized


def unexpected_fields(payload: dict[str, Any]) -> list[str]:
    """Return the keys the model sent that the contract does not cover.

    LLMTriageOutput drops these instead of rejecting them, so this is what keeps
    the dropping observable. All 8 Phase 0 responses included `escalate`, so a
    non-empty result is expected -- and how often it happens is worth reporting.
    """
    return sorted(set(payload) - set(LLMTriageOutput.model_fields))


# Long enough to identify what was wrong, short enough to bound how much
# attacker-controlled text can be replayed into the next prompt.
_MAX_ECHOED_VALUE = 60


def _echo(value: Any) -> str:
    """Render a rejected value for the repair prompt: collapsed, cut, quoted.

    The value is untrusted -- it may be text the model copied out of a ticket,
    including an injection. Whitespace is collapsed first so an embedded newline
    cannot forge extra lines in the correction message. Truncation bounds the
    replay, and the quotes mark it as data rather than instruction.
    """
    if value is None:
        # The model speaks JSON, so a rejected null has to read back as null
        # rather than as Python's None. This is the only Python-to-JSON mapping
        # here: translating anything further would mean inventing a JSON form
        # for a value that may not have one. Unquoted, because it is the JSON
        # literal, not the string "null".
        return "null"
    text = " ".join(str(value).split())
    if len(text) > _MAX_ECHOED_VALUE:
        # Ellipsis so the message never implies the value was short.
        text = text[:_MAX_ECHOED_VALUE] + "..."
    return f'"{text}"'


def correction_message(exc: ValidationError) -> str:
    """Turn a ValidationError into a correction for the Phase 5 repair prompt.

    Names each rejected field, the value that was rejected, and what is allowed
    -- nothing more. Pydantic's own `msg` already states what is allowed
    ("Input should be 'billing', 'bug', ... or 'other'"), so there is no
    per-error-type branching to maintain here.
    """
    lines = []
    for error in exc.errors():
        field = ".".join(str(part) for part in error["loc"]) or "(root)"
        if error["type"] == "missing":
            # For a missing field pydantic sets `input` to the ENTIRE payload,
            # not to the absent value. Echoing it would claim a value was
            # rejected that was never sent, hand the model a Python dict repr
            # when what it needs is "you left out a field", and replay the whole
            # payload into the next prompt -- the exact thing the echo bound
            # exists to prevent. This is the only error type special-cased:
            # everywhere else pydantic's msg already states what is allowed.
            lines.append(f"{field}: {error['msg']}")
        else:
            lines.append(
                f"{field}: rejected {_echo(error.get('input'))} - {error['msg']}"
            )
    return "\n".join(lines)
