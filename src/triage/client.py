"""Single Gemini call: structured output, usage mapping, blocked-response
detection. No retry logic here — SDK/network exceptions propagate to the
caller (service.py), which classifies and retries. A blocked/empty response
is NOT an exception (it's a 200), so it's handled inline and returned as
`RawTriageResponse.error`.
"""
from __future__ import annotations

from dataclasses import dataclass

from config import get_settings
from errors import classify, from_blocked
from google import genai
from google.genai import types
from models import Action, ErrorInfo, Severity, System, TokenUsage, TriageDecision
from pydantic import ValidationError

__all__ = ["DEFAULT_SYSTEM_PROMPT", "RawTriageResponse", "call_gemini", "call_gemini_async", "get_client"]

DEFAULT_SYSTEM_PROMPT = """\
You are an automated cloud server incident triage engine.
Classify the incident into a System, a Severity, and an Action:
  System:   Database, Frontend, Network, or Unknown
  Severity: High, Medium, or Low
  Action:   Restart, Scale, Patch, or Investigate
Return only the structured fields requested. Do not add commentary.
"""

# Gemini's response_schema is a constrained OpenAPI-3.0 subset, NOT full JSON
# Schema - it has no `additionalProperties` support, so TriageDecision's own
# model_json_schema() (which carries `additionalProperties: false` from
# extra="forbid") is rejected outright with a 400. Built by hand instead,
# directly from the enums so it can't drift from models.py. Confirmed live
# against the API - see client.py's history / the session that added this.
_TRIAGE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "system": types.Schema(type=types.Type.STRING, enum=[e.value for e in System]),
        "severity": types.Schema(type=types.Type.STRING, enum=[e.value for e in Severity]),
        "action": types.Schema(type=types.Type.STRING, enum=[e.value for e in Action]),
        "confidence": types.Schema(type=types.Type.NUMBER, nullable=True),
    },
    required=["system", "severity", "action", "confidence"],
)


@dataclass(frozen=True)
class RawTriageResponse:
    """Result of one Gemini call. Exactly one of decision/error is set."""

    decision: TriageDecision | None
    usage: TokenUsage
    raw_text: str | None
    error: ErrorInfo | None


_client: genai.Client | None = None


def get_client() -> genai.Client:
    """Lazily construct the module-level Gemini client.

    Reads GEMINI_API_KEY / GOOGLE_API_KEY from the environment (Option A —
    the credential is never modelled in Settings). Timeout and retry come
    from Settings so they're centrally configurable.
    """
    global _client
    if _client is None:
        settings = get_settings()
        retry_options = None
        if settings.sdk_max_retries > 0:
            retry_options = types.HttpRetryOptions(
                attempts=settings.sdk_max_retries,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            )
        _client = genai.Client(
            http_options=types.HttpOptions(
                timeout=int(settings.request_timeout_s * 1000),  # ms
                retry_options=retry_options,
            )
        )
    return _client


def reset_client() -> None:
    """Test hook: drop the cached client so the next call rebuilds it."""
    global _client
    _client = None


def _build_config(settings) -> types.GenerateContentConfig:
    system_prompt = settings.system_prompt_override() or DEFAULT_SYSTEM_PROMPT
    return types.GenerateContentConfig(
        system_instruction=system_prompt,
        response_mime_type="application/json",
        response_schema=_TRIAGE_SCHEMA,
        max_output_tokens=settings.max_tokens,
        # 2.5 Flash's "thinking" is on by default and is billed out of
        # max_output_tokens - at a small budget like ours it can consume the
        # whole thing and leave the actual JSON answer truncated. This is a
        # short classification task with no real reasoning to do, so disable
        # it outright rather than inflating max_tokens to compensate.
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )


def _usage_from_response(response: types.GenerateContentResponse) -> TokenUsage:
    meta = response.usage_metadata
    if meta is None:
        return TokenUsage(input_tokens=0, output_tokens=0)
    return TokenUsage(
        input_tokens=meta.prompt_token_count or 0,
        output_tokens=meta.candidates_token_count or 0,
        cache_read_input_tokens=meta.cached_content_token_count or 0,
    )


def _safe_text(response: types.GenerateContentResponse) -> str | None:
    try:
        return response.text
    except Exception:  # noqa: BLE001 - .text's failure modes aren't documented; treat any as "no text"
        # .text raises when there is no text part (e.g. fully blocked)
        return None


def _blocked_reasons(response: types.GenerateContentResponse) -> tuple[str | None, str | None]:
    block_reason: str | None = None
    if response.prompt_feedback is not None and response.prompt_feedback.block_reason is not None:
        block_reason = str(response.prompt_feedback.block_reason)

    finish_reason: str | None = None
    if response.candidates:
        fr = response.candidates[0].finish_reason
        finish_reason = str(fr) if fr is not None else None

    return finish_reason, block_reason


def _interpret(response: types.GenerateContentResponse) -> RawTriageResponse:
    """Shared post-processing for the sync and async call paths.

    response.parsed is a plain dict here (we hand Gemini a types.Schema, not
    the Pydantic class, so the SDK has no type to instantiate) - validate it
    into TriageDecision ourselves. Falls back to parsing raw_text in case
    .parsed is ever absent despite valid JSON text.
    """
    usage = _usage_from_response(response)
    raw_text = _safe_text(response)
    decision: TriageDecision | None = None

    try:
        if isinstance(response.parsed, TriageDecision):
            decision = response.parsed
        elif isinstance(response.parsed, dict):
            decision = TriageDecision.model_validate(response.parsed)
        elif raw_text:
            decision = TriageDecision.model_validate_json(raw_text)
    except ValidationError as exc:
        return RawTriageResponse(decision=None, usage=usage, raw_text=raw_text, error=classify(exc))

    if decision is None:
        finish_reason, block_reason = _blocked_reasons(response)
        return RawTriageResponse(
            decision=None,
            usage=usage,
            raw_text=raw_text,
            error=from_blocked(finish_reason, block_reason),
        )

    return RawTriageResponse(decision=decision, usage=usage, raw_text=raw_text, error=None)


def call_gemini(complaint: str) -> RawTriageResponse:
    """One synchronous Gemini call. Raises the underlying SDK/network
    exception on failure — the caller classifies and decides whether to retry.
    """
    settings = get_settings()
    client = get_client()
    response = client.models.generate_content(
        model=settings.model,
        contents=complaint,
        config=_build_config(settings),
    )
    return _interpret(response)


async def call_gemini_async(complaint: str) -> RawTriageResponse:
    """Async counterpart, used by the batch path."""
    settings = get_settings()
    client = get_client()
    response = await client.aio.models.generate_content(
        model=settings.model,
        contents=complaint,
        config=_build_config(settings),
    )
    return _interpret(response)
