"""Application settings, read from the environment.

Two properties this module is built around:

1. No connection value is defaulted. LLM_BASE_URL, LLM_MODEL and LLM_API_KEY
   have no fallback, so an endpoint, a model name, or a credential cannot be
   embedded in tracked source -- pointing this at a different provider is a
   .env edit, not a code change.
2. Importing this module is side-effect free. Nothing reads the environment
   until get_settings() is called, so tests that mock the LLM need no .env.
"""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated settings, sourced from the process environment or a .env file.

    A missing required variable raises ValidationError naming every missing
    field at once, before any ticket is read or any HTTP call is made.
    """

    model_config = SettingsConfigDict(
        # Relative to the working directory, so run the app from the repo root.
        env_file=".env",
        env_file_encoding="utf-8",
        # Already the pydantic-settings default. Stated explicitly because the
        # behaviour is load-bearing: a misspelled key in .env (LLM_MODEL_NAME
        # instead of LLM_MODEL) raises rather than being silently ignored,
        # which would otherwise leave the real setting mysteriously unset.
        extra="forbid",
    )

    # -- Required. A default for any of these would be a hardcoded connection --
    # -- value, and min_length=1 stops a copied-but-unfilled .env from passing --
    # -- an empty string through to the client, where it would fail obscurely. --

    llm_base_url: str = Field(min_length=1)
    llm_model: str = Field(min_length=1)

    # SecretStr masks the value in repr(), so printing or logging this object,
    # or a ValidationError echoing it, cannot leak the key. Ollama ignores the
    # value, but the openai SDK will not construct a client without one -- and
    # it is a live credential as soon as llm_base_url is a hosted provider.
    # Costs one explicit .get_secret_value() at the single point of use.
    llm_api_key: SecretStr = Field(min_length=1)

    # -- Operational knobs, defaulted from the Phase 0 timing run. --

    # The first call after model load took 17.3s; steady state was 2.7-7.0s.
    # Any timeout under ~20s would make the first ticket of every batch time
    # out, be treated as a transport failure, and fall back -- a failure caused
    # entirely by our own configuration rather than by the model.
    llm_timeout_seconds: float = Field(default=60.0, gt=0)

    # The longest Phase 0 response -- preamble, fenced JSON, trailing paragraph
    # -- ran to roughly 260 tokens. Truncation sets finish_reason='length' and
    # produces unparseable output, so this sits at about 2x what was observed
    # while still bounding a runaway generation.
    llm_max_tokens: int = Field(default=512, gt=0)

    # Retries AFTER the first attempt: 2 means at most 3 calls. The cap of 3
    # is derived from measurement rather than picked. Phase 0 steady-state
    # calls averaged 5.3s (seven runs, 2.7-7.0s), so a full 30-ticket pass
    # costs ~2.6 min at 1 attempt, ~5.3 min at 2, and ~7.9 min at 3. The
    # prompt is expected to go through roughly ten eval runs while it is
    # iterated on, which makes ~8 minutes per run the ceiling worth paying;
    # 3 attempts sits at that ceiling. 0 disables retrying while iterating.
    llm_max_retries: int = Field(default=2, ge=0, le=3)

    # -- Policy. --

    # Final confidence below which a ticket is escalated to a human. The
    # comparison is strict: escalate when final_confidence < threshold, so a
    # ticket landing exactly on the threshold is not escalated. The operator is
    # named here so the policy code and this comment cannot drift apart. No
    # default: which tickets reach a person is a policy decision that has to be
    # made deliberately, and a silent default would mean shipping a threshold
    # nobody chose. The 0-1 bound is the same constraint a Phase 0 response
    # reporting confidence: 100 violated.
    confidence_threshold: float = Field(ge=0.0, le=1.0)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings, reading the environment once.

    Cached so the .env is parsed a single time per process. A test can swap the
    environment with monkeypatch.setenv(...) then get_settings.cache_clear().
    """
    return Settings()
