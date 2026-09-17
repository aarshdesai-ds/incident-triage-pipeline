"""Unit tests for service.py (retry orchestration). client.call_gemini* is
mocked - no network / API key needed. Run: pytest src/triage/test_service.py -v
"""
from __future__ import annotations

import pytest
import service as service_mod
from client import RawTriageResponse
from config import get_settings
from errors import from_blocked
from google.genai import errors as genai_errors
from models import (
    Action,
    Severity,
    System,
    TokenUsage,
    TriageDecision,
    TriageErrorType,
    TriageRequest,
)


@pytest.fixture(autouse=True)
def _reset_settings(monkeypatch):
    monkeypatch.setenv("TRIAGE_APP_MAX_ATTEMPTS", "3")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    # keep retry tests fast
    monkeypatch.setattr(service_mod.time, "sleep", lambda _s: None)
    monkeypatch.setattr(service_mod, "_backoff_seconds", lambda _a: 0)

    async def _fast_asleep(_s):
        return None

    monkeypatch.setattr(service_mod.asyncio, "sleep", _fast_asleep)


def _decision() -> TriageDecision:
    return TriageDecision(system=System.DATABASE, severity=Severity.HIGH, action=Action.RESTART, confidence=0.8)


def _request() -> TriageRequest:
    return TriageRequest(incident_id="INC-1", complaint="db is on fire")


class TestTriageSuccess:
    def test_success_first_try(self, monkeypatch):
        raw = RawTriageResponse(decision=_decision(), usage=TokenUsage(input_tokens=10, output_tokens=5), raw_text="{}", error=None)
        monkeypatch.setattr(service_mod, "call_gemini", lambda complaint: raw)

        result = service_mod.triage(_request())
        assert result.status == "success"
        assert result.decision == _decision()
        assert result.attempts == 1
        assert result.usage.total_tokens == 15
        assert result.latency_ms >= 0


class TestTriageRetryableFailure:
    def test_retries_then_succeeds(self, monkeypatch):
        calls = {"n": 0}

        def flaky(complaint):
            calls["n"] += 1
            if calls["n"] < 3:
                raise genai_errors.ServerError(503, {"message": "overloaded"})
            return RawTriageResponse(decision=_decision(), usage=TokenUsage(input_tokens=1, output_tokens=1), raw_text="{}", error=None)

        monkeypatch.setattr(service_mod, "call_gemini", flaky)
        result = service_mod.triage(_request())

        assert result.status == "success"
        assert result.attempts == 3
        assert calls["n"] == 3

    def test_exhausts_retries_and_fails(self, monkeypatch):
        def always_fails(complaint):
            raise genai_errors.ServerError(500, {"message": "down"})

        monkeypatch.setattr(service_mod, "call_gemini", always_fails)
        result = service_mod.triage(_request())

        assert result.status == "error"
        assert result.error.type is TriageErrorType.SERVER_ERROR
        assert result.attempts == 3  # TRIAGE_APP_MAX_ATTEMPTS


class TestTriageNonRetryableFailure:
    def test_bad_request_fails_immediately(self, monkeypatch):
        calls = {"n": 0}

        def bad_request(complaint):
            calls["n"] += 1
            raise genai_errors.ClientError(400, {"message": "bad model"})

        monkeypatch.setattr(service_mod, "call_gemini", bad_request)
        result = service_mod.triage(_request())

        assert result.status == "error"
        assert result.error.type is TriageErrorType.BAD_REQUEST
        assert result.attempts == 1
        assert calls["n"] == 1  # never retried


class TestTriageBlockedResponse:
    def test_blocked_not_retried(self, monkeypatch):
        calls = {"n": 0}
        blocked = RawTriageResponse(decision=None, usage=TokenUsage(input_tokens=5, output_tokens=0), raw_text=None, error=from_blocked("SAFETY"))

        def call(complaint):
            calls["n"] += 1
            return blocked

        monkeypatch.setattr(service_mod, "call_gemini", call)
        result = service_mod.triage(_request())

        assert result.status == "error"
        assert result.error.type is TriageErrorType.REFUSAL
        assert calls["n"] == 1


class TestTriageAsync:
    @pytest.mark.asyncio
    async def test_success(self, monkeypatch):
        async def call(complaint):
            return RawTriageResponse(decision=_decision(), usage=TokenUsage(input_tokens=2, output_tokens=2), raw_text="{}", error=None)

        monkeypatch.setattr(service_mod, "call_gemini_async", call)
        result = await service_mod.triage_async(_request())
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_retries_then_fails(self, monkeypatch):
        calls = {"n": 0}

        async def call(complaint):
            calls["n"] += 1
            raise genai_errors.ClientError(429, {"message": "slow down"})

        monkeypatch.setattr(service_mod, "call_gemini_async", call)
        result = await service_mod.triage_async(_request())

        assert result.status == "error"
        assert result.error.type is TriageErrorType.RATE_LIMIT
        assert calls["n"] == 3
