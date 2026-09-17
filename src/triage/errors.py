"""Exception / blocked-response -> ErrorInfo mapping (Gemini).

Pure functions. `service.py` calls `classify()` around the Gemini call and
`from_blocked()` when a 200 response carries no usable content (safety /
recitation / other stop reasons).
"""
from __future__ import annotations

import httpx
from google.genai import errors as genai_errors
from models import ErrorInfo, TriageErrorType
from pydantic import ValidationError

__all__ = ["RETRYABLE_TYPES", "classify", "from_blocked"]

_MAX_MESSAGE_LEN = 500

RETRYABLE_TYPES: frozenset[TriageErrorType] = frozenset(
    {
        TriageErrorType.RATE_LIMIT,
        TriageErrorType.SERVER_ERROR,
        TriageErrorType.CONNECTION,
        TriageErrorType.TIMEOUT,
    }
)


def _message(exc: Exception) -> str:
    raw = getattr(exc, "message", None) or str(exc) or exc.__class__.__name__
    return raw[:_MAX_MESSAGE_LEN]


def _info(type_: TriageErrorType, exc: Exception) -> ErrorInfo:
    return ErrorInfo(
        type=type_,
        message=_message(exc),
        retryable=type_ in RETRYABLE_TYPES,
        request_id=None,  # Gemini responses carry no request id
    )


def _classify_client_error(exc: genai_errors.ClientError) -> ErrorInfo:
    """Gemini lumps every 4xx into ClientError; branch on exc.code."""
    code = exc.code
    if code == 429:
        return _info(TriageErrorType.RATE_LIMIT, exc)
    if code in (401, 403):
        return _info(TriageErrorType.AUTHENTICATION, exc)
    if code == 408:
        return _info(TriageErrorType.TIMEOUT, exc)
    return _info(TriageErrorType.BAD_REQUEST, exc)  # 400, 404, 413, 422, ...


def classify(exc: Exception) -> ErrorInfo:
    """Map any exception to an ErrorInfo. Never raises."""
    if isinstance(exc, ValidationError):
        return _info(TriageErrorType.SCHEMA_VALIDATION, exc)

    # network failures surface as plain httpx exceptions (google-genai's
    # transport dependency); check timeout before the broader TransportError
    if isinstance(exc, httpx.TimeoutException):
        return _info(TriageErrorType.TIMEOUT, exc)
    if isinstance(exc, httpx.TransportError):
        return _info(TriageErrorType.CONNECTION, exc)

    if isinstance(exc, genai_errors.ClientError):
        return _classify_client_error(exc)
    if isinstance(exc, genai_errors.ServerError):
        return _info(TriageErrorType.SERVER_ERROR, exc)
    if isinstance(exc, genai_errors.UnknownApiResponseError):
        return _info(TriageErrorType.SERVER_ERROR, exc)
    if isinstance(exc, genai_errors.APIError):
        return _info(TriageErrorType.UNKNOWN, exc)

    # not a Gemini/network error at all - a bug in our own code
    return ErrorInfo(
        type=TriageErrorType.UNKNOWN,
        message=_message(exc),
        retryable=False,
        request_id=None,
    )


def from_blocked(finish_reason: str | None, block_reason: str | None = None) -> ErrorInfo:
    """ErrorInfo for a 200 response with no usable content.

    `block_reason` (from `prompt_feedback`) means the *input* was blocked;
    `finish_reason` (from the first candidate) means generation stopped
    early (SAFETY, RECITATION, PROHIBITED_CONTENT, MALFORMED_FUNCTION_CALL,
    OTHER, ...). Neither case is retried.
    """
    if block_reason:
        message = f"prompt blocked: {block_reason}"
    else:
        message = f"generation stopped: {finish_reason or 'unknown'}"
    return ErrorInfo(
        type=TriageErrorType.REFUSAL,
        message=message[:_MAX_MESSAGE_LEN],
        retryable=False,
        request_id=None,
    )
