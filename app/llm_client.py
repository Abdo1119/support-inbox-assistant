"""Transport for the LLM call. Raw text in, raw text out, nothing interpreted.

This layer answers exactly one question: did the bytes arrive? It returns the
content the model produced plus the metadata around the call, and classifies a
failure without deciding what to do about it. It does not parse, validate, or
judge whether the text is usable -- those are Phase 5's questions, and keeping
them out of here is what lets the reliability layer be tested against a mocked
client with no live model.

Two budgets exist in this system and they must add rather than multiply:

  transport retry  (here)     the SAME messages re-sent, because nothing arrived
  repair retry     (Phase 5)  DIFFERENT messages, because what arrived was invalid

Nesting them would give (N+1)^2 calls per ticket -- 9 at N=2. So the caller owns
a single per-ticket budget: it passes `max_attempts`, reads `attempts` back, and
spends what is left on a repair. Total calls per ticket stay within
LLM_MAX_RETRIES + 1, which is the ceiling CLAUDE.md's timing arithmetic sets.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache

import openai
from openai import OpenAI

from app.config import get_settings

logger = logging.getLogger(__name__)


class FailureClass(str, Enum):
    """Why a call failed, and implicitly whether trying again could help."""

    # Retryable: the condition is external and may not hold on the next attempt.
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    SERVER_ERROR = "server_error"
    RATE_LIMIT = "rate_limit"

    # Non-retryable: deterministic. The same request fails the same way, so
    # spending the budget on it buys no chance of success.
    NOT_FOUND = "not_found"
    BAD_REQUEST = "bad_request"
    AUTH = "auth"
    EMPTY_MESSAGES = "empty_messages"


_RETRYABLE = frozenset(
    {
        FailureClass.TIMEOUT,
        FailureClass.CONNECTION,
        FailureClass.SERVER_ERROR,
        FailureClass.RATE_LIMIT,
    }
)


@dataclass(frozen=True)
class LLMCallResult:
    """One call attempt sequence, reported rather than acted upon.

    A frozen dataclass rather than a Pydantic model: every field is constructed
    by this module, so there is no untrusted input to coerce or validate.
    Pydantic earns its place where the model's own output arrives, in Phase 5.
    """

    ok: bool
    # Optional by contract: the SDK types message.content as Optional[str], so
    # ok=True with content=None is representable and correct here -- transport
    # succeeded, there is simply no text. Phase 5 treats it as a syntactic
    # failure, since there is nothing to extract from.
    content: str | None
    finish_reason: str | None
    truncated: bool
    # Wall clock for the WHOLE sequence: every attempt plus the backoff sleeps
    # between them.
    elapsed_seconds: float
    # Wall clock for the FINAL attempt alone. Averaging elapsed_seconds across
    # tickets inflates the mean wherever a retry happened; this is the figure
    # that belongs in a latency table.
    last_attempt_seconds: float
    prompt_tokens: int | None
    completion_tokens: int | None
    attempts: int
    failure: FailureClass | None = None
    retryable: bool = False
    error_detail: str | None = None


@lru_cache
def get_client() -> OpenAI:
    """Build the OpenAI client from settings. Cached: one client per process."""
    settings = get_settings()
    return OpenAI(
        base_url=settings.llm_base_url,
        # The only place the secret is unwrapped. SecretStr keeps it out of
        # reprs and logs everywhere else.
        api_key=settings.llm_api_key.get_secret_value(),
        timeout=settings.llm_timeout_seconds,
        # The SDK defaults this to 2. Left alone it would multiply against
        # LLM_MAX_RETRIES for up to 9 calls per ticket, and the extra attempts
        # would be invisible -- they happen inside the SDK and never reach our
        # logs. Our retry policy has to be the only one.
        max_retries=0,
    )


def _classify(exc: Exception) -> FailureClass:
    """Map an SDK exception to a failure class.

    APITimeoutError is checked before APIConnectionError because it subclasses
    it; the reverse order would swallow every timeout into CONNECTION.
    """
    if isinstance(exc, openai.APITimeoutError):
        return FailureClass.TIMEOUT
    if isinstance(exc, openai.APIConnectionError):
        return FailureClass.CONNECTION
    if isinstance(exc, openai.RateLimitError):
        return FailureClass.RATE_LIMIT
    if isinstance(exc, openai.NotFoundError):
        # Ollama answers 404 for an unknown model tag.
        return FailureClass.NOT_FOUND
    if isinstance(exc, (openai.AuthenticationError, openai.PermissionDeniedError)):
        return FailureClass.AUTH
    if isinstance(exc, openai.InternalServerError):
        return FailureClass.SERVER_ERROR
    if isinstance(exc, openai.APIStatusError):
        # Anything not named above: 5xx is worth another try, 4xx is not.
        return (
            FailureClass.SERVER_ERROR
            if exc.status_code >= 500
            else FailureClass.BAD_REQUEST
        )
    # An unrecognised exception is treated as non-retryable. Retrying something
    # we cannot classify is how a bug turns into three copies of itself.
    return FailureClass.BAD_REQUEST


def call_llm(
    messages: list[dict[str, str]],
    *,
    max_attempts: int | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> LLMCallResult:
    """Send `messages` and return the raw response with its metadata.

    `max_attempts` is the caller's remaining per-ticket budget, defaulting to
    LLM_MAX_RETRIES + 1. Only TRANSPORT failures are retried here -- this layer
    cannot see a schema failure and must not pretend to.

    `temperature` defaults to 0.0 and is a parameter so Phase 5 can run the
    self-consistency check on ambiguous tickets without a second function.
    """
    settings = get_settings()
    if max_attempts is None:
        max_attempts = settings.llm_max_retries + 1
    if max_tokens is None:
        max_tokens = settings.llm_max_tokens

    if not messages:
        # Deterministic, and no call is made at all -- attempts stays 0.
        logger.error("llm call refused: empty message list")
        return LLMCallResult(
            ok=False,
            content=None,
            finish_reason=None,
            truncated=False,
            elapsed_seconds=0.0,
            last_attempt_seconds=0.0,
            prompt_tokens=None,
            completion_tokens=None,
            attempts=0,
            failure=FailureClass.EMPTY_MESSAGES,
            retryable=False,
            error_detail="messages list is empty",
        )

    # Ticket text is never logged. Shape only.
    logger.info(
        "llm call: model=%s messages=%d chars=%d max_attempts=%d",
        settings.llm_model,
        len(messages),
        sum(len(m.get("content", "")) for m in messages),
        max_attempts,
    )

    client = get_client()
    started = time.monotonic()
    attempts = 0
    last_attempt_seconds = 0.0
    failure: FailureClass = FailureClass.CONNECTION
    detail = ""

    while attempts < max_attempts:
        attempts += 1
        attempt_started = time.monotonic()
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                # Per-request, so the model stays resident without depending on
                # an environment variable that needs an Ollama restart to
                # change. Measured: a cold reload costs 17.3s against a 5.3s
                # warm call, and the default idle unload is 5 minutes.
                extra_body={"keep_alive": "30m"},
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a classified result
            last_attempt_seconds = time.monotonic() - attempt_started
            failure = _classify(exc)
            detail = f"{type(exc).__name__}: {exc}"
            retryable = failure in _RETRYABLE
            logger.warning(
                "llm attempt %d/%d failed in %.1fs: failure=%s retryable=%s %s",
                attempts,
                max_attempts,
                last_attempt_seconds,
                failure.value,
                retryable,
                type(exc).__name__,
            )
            if not retryable:
                break
            if attempts < max_attempts:
                backoff = 2 ** (attempts - 1)
                logger.info("backing off %ds before retry", backoff)
                time.sleep(backoff)
            continue

        last_attempt_seconds = time.monotonic() - attempt_started
        elapsed = time.monotonic() - started
        choice = response.choices[0]
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else None
        completion_tokens = usage.completion_tokens if usage else None
        truncated = choice.finish_reason == "length"

        if truncated:
            # Our configuration failing, not the model misbehaving: the response
            # hit max_tokens and stopped mid-output. Distinct from a natural
            # stop, and loud, because re-sending produces the same truncation.
            logger.warning(
                "llm attempt %d/%d ok in %.1fs (sequence %.1fs) but TRUNCATED at "
                "max_tokens=%d: finish_reason=length prompt=%s completion=%s",
                attempts,
                max_attempts,
                last_attempt_seconds,
                elapsed,
                max_tokens,
                prompt_tokens,
                completion_tokens,
            )
        else:
            logger.info(
                "llm attempt %d/%d ok in %.1fs (sequence %.1fs): "
                "finish_reason=%s prompt=%s completion=%s",
                attempts,
                max_attempts,
                last_attempt_seconds,
                elapsed,
                choice.finish_reason,
                prompt_tokens,
                completion_tokens,
            )

        return LLMCallResult(
            ok=True,
            content=choice.message.content,
            finish_reason=choice.finish_reason,
            truncated=truncated,
            elapsed_seconds=elapsed,
            last_attempt_seconds=last_attempt_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            attempts=attempts,
        )

    elapsed = time.monotonic() - started
    logger.error(
        "llm call gave up after %d attempt(s), sequence %.1fs "
        "(last attempt %.1fs): failure=%s",
        attempts,
        elapsed,
        last_attempt_seconds,
        failure.value,
    )
    return LLMCallResult(
        ok=False,
        content=None,
        finish_reason=None,
        truncated=False,
        elapsed_seconds=elapsed,
        last_attempt_seconds=last_attempt_seconds,
        prompt_tokens=None,
        completion_tokens=None,
        attempts=attempts,
        failure=failure,
        retryable=failure in _RETRYABLE,
        error_detail=detail,
    )
