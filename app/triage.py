"""Triage one ticket into one TriageRecord.

This module owns the six stations and the policy layer on top of them:

    transport -> extract -> parse -> normalize -> validate -> policy

Three principles run through it, all of them reactions to measured Phase 0
behaviour rather than to imagined failure:

1. **Rule signals lower confidence; they never change the model's category.**
   Two systems overruling each other is worse than one flagging doubt. Every
   penalty here subtracts, and none of them rewrites a decision.
2. **Failure claims less, it does not invent a substitute.** The fallback
   summary is the ticket body truncated, never model output, and never a
   sentence the system made up.
3. **One budget per ticket, shared with transport.** Repair retries and
   transport retries draw from the same pool, so the worst case stays at
   LLM_MAX_RETRIES + 1 calls rather than multiplying to (N+1)^2.
"""

from __future__ import annotations

import json
import logging

from pydantic import ValidationError

from app.config import get_settings
from app.llm_client import LLMCallResult, call_llm
from app.prompts import build_messages
from app.schemas import (
    Category,
    LLMTriageOutput,
    Priority,
    TriageRecord,
    correction_message,
    normalize_llm_payload,
    unexpected_fields,
)

logger = logging.getLogger(__name__)


# --- Policy constants --------------------------------------------------------

# Words that indicate someone REPORTING a security problem. Deliberately
# excludes "password", "login", "secret" and "signature": those are ordinary
# account and bug vocabulary here -- T-005 is a password-reset ticket labeled
# account, T-019 is an HMAC signature failure labeled bug -- and including them
# would fire the conflict penalty on correctly classified tickets.
#
# Verified against all 30 tickets: matches T-014 only, and no labeled ticket.
# Known blind spot: it does not fire on T-015, which is phishing but never says
# so. Keyword matching only catches security problems that describe themselves.
SECURITY_KEYWORDS = (
    "vulnerabilit", "exploit", "breach", "hacked", "phishing", "idor", "xss",
    "csrf", "sql injection", "malware", "ransomware", "compromis",
    "unauthorized", "unauthorised", "data leak", "responsible disclosure",
    "penetration test", "another tenant", "other tenants", "scam", "fraud",
)

# A body shorter than this carries too little to classify. T-018 is six words
# and Phase 0 answered bug/high at 0.8 confidence for it.
MIN_BODY_WORDS = 10

# How much of the body becomes the fallback summary. Truncated, never generated.
FALLBACK_SUMMARY_CHARS = 200

# Penalties. Each one lowers confidence; none touches the category.
PENALTY_SUMMARY_IS_SUBJECT = 0.25
PENALTY_SHORT_BODY = 0.25
PENALTY_SECURITY_CONFLICT = 0.35
PENALTY_REPAIRED = 0.20
PENALTY_TRUNCATED = 0.30

# The model's self-reported number can move the result by at most this much. It
# is an input, never the answer: Phase 0 returned 0.8 for a correct call, a
# wrong call, and a six-word ticket; 0 for the empty body; 100 for the injection.
#
# Why the weight is small rather than zero. On T-004 (gibberish) the model
# reported 0.1, which pulled the blended score DOWN from 0.75 to 0.652 -- the
# model's own doubt moved the result in the right direction. With Phase 0's 0 on
# the empty body, the pattern is that the number carries some signal at the
# extremes and essentially none in the middle, where 0.8 covered a correct call,
# a wrong call, and a six-word ticket alike. Small weight keeps the useful tails
# without letting the uninformative middle decide anything.
MODEL_CONFIDENCE_WEIGHT = 0.15

# Appended once to the repair turn, not repeated per field.
REPAIR_INSTRUCTION = (
    "Your previous response was rejected. Correct only the fields listed above "
    "and reply with the corrected JSON object and nothing else."
)


# --- Station 2: extraction ---------------------------------------------------


def extract_json_object(text: str | None) -> str | None:
    """Pull the JSON object out of whatever the model wrapped it in.

    All 8 Phase 0 responses were prose preamble + a fenced JSON block + a
    trailing paragraph, so this is the normal path rather than an edge case.
    First brace to last brace: simple, and correct for a single top-level
    object, which is all the schema ever asks for.

    Returns None when there is nothing to extract -- including when `text` is
    None, which the transport layer can legitimately return alongside ok=True.
    The caller treats that as a syntactic failure, not a crash.
    """
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    return text[start : end + 1]


# --- Policy: confidence signals ---------------------------------------------


def confidence_signals(
    *,
    subject: str,
    body: str,
    output: LLMTriageOutput,
    repaired: bool,
    truncated: bool,
) -> list[tuple[str, float]]:
    """Return (reason, penalty) pairs for everything working against trust.

    Pure and LLM-free, so the arithmetic is inspectable and testable on its own.
    Every entry lowers confidence -- there are no positive signals, because a
    model that sounds sure is not evidence of anything.
    """
    signals: list[tuple[str, float]] = []

    if output.summary.strip().lower() == subject.strip().lower() and subject.strip():
        # 7 of 8 Phase 0 runs returned the subject verbatim as the summary. It
        # means the body was never read, so the category came from a subject
        # line that is often vague, wrong, or empty.
        signals.append(("summary repeats the subject line", PENALTY_SUMMARY_IS_SUBJECT))

    if len(body.split()) < MIN_BODY_WORDS:
        signals.append(
            (f"body is only {len(body.split())} words", PENALTY_SHORT_BODY)
        )

    text = f"{subject} {body}".lower()
    hits = [kw for kw in SECURITY_KEYWORDS if kw in text]
    if hits and output.category is not Category.SECURITY:
        # The conflict itself is the signal. It never rewrites the category --
        # coercing it would be the system overruling the model on a keyword.
        signals.append(
            (
                f"security keywords {hits[:3]} present but category is "
                f"{output.category.value}",
                PENALTY_SECURITY_CONFLICT,
            )
        )

    if repaired:
        signals.append(("needed a repair retry", PENALTY_REPAIRED))

    if truncated:
        signals.append(("response was truncated", PENALTY_TRUNCATED))

    return signals


def _final_confidence(signals: list[tuple[str, float]], model_confidence: float) -> float:
    """Combine rule penalties with the model's own number, weighted low."""
    rule_score = max(0.0, 1.0 - sum(penalty for _reason, penalty in signals))
    blended = (
        (1.0 - MODEL_CONFIDENCE_WEIGHT) * rule_score
        + MODEL_CONFIDENCE_WEIGHT * model_confidence
    )
    return min(1.0, max(0.0, blended))


def _fallback_record(
    ticket_id: str, body: str, reason: str, retries: int
) -> TriageRecord:
    """Build the record used when no usable model output exists.

    The summary is the ticket body truncated. The model never writes this --
    failure means claiming less, not inventing a substitute. priority=medium
    rather than low because low means "defer" and we do not know that.
    """
    summary = body.strip()[:FALLBACK_SUMMARY_CHARS]
    return TriageRecord(
        ticket_id=ticket_id,
        category=Category.OTHER,
        priority=Priority.MEDIUM,
        summary=summary,
        # No draft reply exists. An empty string states the absence honestly;
        # TriageRecord permits it where LLMTriageOutput would not.
        suggested_reply="",
        model_confidence=0.0,
        final_confidence=0.0,
        escalate=True,
        escalation_reason=reason,
        fallback_used=True,
        retries=retries,
    )


# --- The orchestrator --------------------------------------------------------


def triage_ticket(ticket: dict[str, str]) -> TriageRecord:
    """Run one ticket through all six stations and the policy layer."""
    settings = get_settings()
    ticket_id = ticket["id"]
    subject = ticket.get("subject", "") or ""
    body = ticket.get("body", "") or ""

    # --- Short-circuit: deterministic input failure, no call worth making ----
    if not body.strip():
        # Nothing to put in the data block. Phase 0 showed the model inventing
        # feature_request from the subject line alone on exactly this ticket.
        logger.info("%s short-circuited: empty body, no LLM call", ticket_id)
        return _fallback_record(
            ticket_id, body, "empty ticket body; no LLM call made", retries=0
        )

    # --- One budget for this ticket, shared by transport and repair ----------
    budget = settings.llm_max_retries + 1
    messages = build_messages(subject, body)

    result: LLMCallResult = call_llm(messages, max_attempts=budget)
    budget -= result.attempts
    calls = 1
    repaired = False

    if not result.ok:
        reason = f"transport failure after {result.attempts} attempt(s): {result.failure.value if result.failure else 'unknown'}"
        logger.warning("%s falling back: %s", ticket_id, reason)
        return _fallback_record(ticket_id, body, reason, retries=result.attempts - 1)

    # --- Extract -> parse -> normalize -> validate, repairing while budget --
    output: LLMTriageOutput | None = None
    last_problem = "no parseable JSON in the response"

    while True:
        raw = extract_json_object(result.content)
        if raw is None:
            last_problem = "no JSON object found in the response"
            exc: ValidationError | None = None
        else:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as e:
                last_problem = f"JSON parse failed: {e}"
                exc = None
            else:
                if not isinstance(payload, dict):
                    last_problem = f"top-level JSON is {type(payload).__name__}, not an object"
                    exc = None
                else:
                    extras = unexpected_fields(payload)
                    if extras:
                        # Dropped, not rejected -- but recorded, so the habit
                        # stays measurable. 8 of 8 Phase 0 responses did this.
                        logger.info("%s model sent extra fields: %s", ticket_id, extras)
                    try:
                        output = LLMTriageOutput(**normalize_llm_payload(payload))
                        break
                    except ValidationError as e:
                        exc = e
                        last_problem = "schema validation failed"

        # A truncated response is a deterministic failure: the same max_tokens
        # cuts the same response at the same place, so a repair cannot succeed.
        if result.truncated:
            logger.warning(
                "%s truncated response is not repairable (max_tokens); "
                "falling back", ticket_id
            )
            return _fallback_record(
                ticket_id,
                body,
                f"response truncated at max_tokens; {last_problem}",
                retries=calls - 1,
            )

        if budget <= 0:
            logger.warning("%s budget exhausted after %d call(s): %s", ticket_id, calls, last_problem)
            return _fallback_record(
                ticket_id, body, f"{last_problem}; budget exhausted", retries=calls - 1
            )

        # Build the repair turn: the model's own bad output back as an
        # assistant message, then the correction plus ONE instruction.
        correction = correction_message(exc) if exc is not None else last_problem
        messages = messages + [
            {"role": "assistant", "content": result.content or ""},
            {"role": "user", "content": f"{correction}\n\n{REPAIR_INSTRUCTION}"},
        ]
        logger.info("%s repairing (%s), budget left %d", ticket_id, last_problem, budget)

        result = call_llm(messages, max_attempts=budget)
        budget -= result.attempts
        calls += 1
        repaired = True

        if not result.ok:
            reason = f"transport failure during repair: {result.failure.value if result.failure else 'unknown'}"
            logger.warning("%s falling back: %s", ticket_id, reason)
            return _fallback_record(ticket_id, body, reason, retries=calls - 1)

    # --- Policy -------------------------------------------------------------
    signals = confidence_signals(
        subject=subject,
        body=body,
        output=output,
        repaired=repaired,
        truncated=result.truncated,
    )
    final_confidence = _final_confidence(signals, output.confidence)

    # Everything that fired, kept regardless of whether it escalates. A ticket
    # can carry doubt and still clear the threshold, and that observation is
    # worth recording rather than discarding.
    fired = [reason for reason, _penalty in signals]

    reasons = list(fired)

    # Hard triggers escalate regardless of the score.
    if output.category is Category.SECURITY:
        # Zero labeled security examples means accuracy says nothing about the
        # category where a miss costs most. Never auto-clear one.
        reasons.append("category is security; always reviewed by a human")
    if result.truncated:
        reasons.append("response was truncated")

    hard_trigger = output.category is Category.SECURITY or result.truncated
    below_threshold = final_confidence < settings.confidence_threshold
    escalate = hard_trigger or below_threshold

    if below_threshold:
        # Whenever the score crossed the line, say so with the numbers. Signal
        # names alone tell a reviewer what was noticed but not that it was
        # enough to cross -- and on a ticket with no signals at all, the score
        # is the only reason there is.
        reasons.append(
            f"final confidence {final_confidence:.2f} below threshold "
            f"{settings.confidence_threshold:.2f}"
        )

    logger.info(
        "%s -> %s/%s model_conf=%.2f final_conf=%.2f calls=%d escalate=%s signals=%s",
        ticket_id,
        output.category.value,
        output.priority.value,
        output.confidence,
        final_confidence,
        calls,
        escalate,
        [f"{r} (-{p})" for r, p in signals],
    )

    return TriageRecord(
        ticket_id=ticket_id,
        category=output.category,
        priority=output.priority,
        summary=output.summary,
        suggested_reply=output.suggested_reply,
        model_confidence=output.confidence,
        final_confidence=final_confidence,
        confidence_signals=fired,
        escalate=escalate,
        escalation_reason="; ".join(reasons) if escalate else None,
        fallback_used=False,
        retries=calls - 1,
    )
