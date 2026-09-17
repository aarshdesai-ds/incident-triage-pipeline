"""Retry orchestration + TriageResult assembly.

`client.py` makes one call and either returns a RawTriageResponse or lets an
exception propagate. This module owns the retry loop (`settings.app_max_attempts`),
turns exceptions into ErrorInfo via `errors.classify`, and assembles the final
TriageResult envelope.
"""
from __future__ import annotations

import asyncio
import time

from client import call_gemini, call_gemini_async
from config import get_settings
from errors import classify
from models import TokenUsage, TriageRequest, TriageResult

__all__ = ["triage", "triage_async"]

ZERO_USAGE = TokenUsage(input_tokens=0, output_tokens=0)
_MAX_BACKOFF_S = 10.0


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff, capped. attempt is 1-indexed (the attempt that just failed)."""
    return min(2 ** (attempt - 1), _MAX_BACKOFF_S)


def _elapsed_ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)


def triage(request: TriageRequest) -> TriageResult:
    """Synchronous single-incident triage with app-level retry."""
    settings = get_settings()
    start = time.monotonic()
    attempts = 0
    last_usage = ZERO_USAGE

    while True:
        attempts += 1
        try:
            raw = call_gemini(request.complaint)
        except Exception as exc:  # noqa: BLE001 - deliberately broad; classified below
            error = classify(exc)
            if error.retryable and attempts < settings.app_max_attempts:
                time.sleep(_backoff_seconds(attempts))
                continue
            return TriageResult(
                incident_id=request.incident_id,
                complaint=request.complaint,
                status="error",
                error=error,
                usage=last_usage,
                model=settings.model,
                attempts=attempts,
                latency_ms=_elapsed_ms(start),
            )

        last_usage = raw.usage

        if raw.error is not None:
            # blocked / unparseable response - not retried
            return TriageResult(
                incident_id=request.incident_id,
                complaint=request.complaint,
                status="error",
                error=raw.error,
                raw_response=raw.raw_text,
                usage=raw.usage,
                model=settings.model,
                attempts=attempts,
                latency_ms=_elapsed_ms(start),
            )

        return TriageResult(
            incident_id=request.incident_id,
            complaint=request.complaint,
            status="success",
            decision=raw.decision,
            raw_response=raw.raw_text,
            usage=raw.usage,
            model=settings.model,
            attempts=attempts,
            latency_ms=_elapsed_ms(start),
        )


async def triage_async(request: TriageRequest) -> TriageResult:
    """Async counterpart, used by the batch path and the FastAPI endpoint."""
    settings = get_settings()
    start = time.monotonic()
    attempts = 0
    last_usage = ZERO_USAGE

    while True:
        attempts += 1
        try:
            raw = await call_gemini_async(request.complaint)
        except Exception as exc:  # noqa: BLE001
            error = classify(exc)
            if error.retryable and attempts < settings.app_max_attempts:
                await asyncio.sleep(_backoff_seconds(attempts))
                continue
            return TriageResult(
                incident_id=request.incident_id,
                complaint=request.complaint,
                status="error",
                error=error,
                usage=last_usage,
                model=settings.model,
                attempts=attempts,
                latency_ms=_elapsed_ms(start),
            )

        last_usage = raw.usage

        if raw.error is not None:
            return TriageResult(
                incident_id=request.incident_id,
                complaint=request.complaint,
                status="error",
                error=raw.error,
                raw_response=raw.raw_text,
                usage=raw.usage,
                model=settings.model,
                attempts=attempts,
                latency_ms=_elapsed_ms(start),
            )

        return TriageResult(
            incident_id=request.incident_id,
            complaint=request.complaint,
            status="success",
            decision=raw.decision,
            raw_response=raw.raw_text,
            usage=raw.usage,
            model=settings.model,
            attempts=attempts,
            latency_ms=_elapsed_ms(start),
        )
