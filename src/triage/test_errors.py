"""Unit tests for errors.py.  Run: pytest src/triage/test_errors.py -v"""
from __future__ import annotations

import httpx
import pytest
from errors import RETRYABLE_TYPES, classify, from_blocked
from google.genai import errors as genai_errors
from models import TriageErrorType
from pydantic import ValidationError


def _client_error(code: int, message: str = "boom") -> genai_errors.ClientError:
    return genai_errors.ClientError(code, {"message": message, "status": "X"})


def _server_error(code: int, message: str = "down") -> genai_errors.ServerError:
    return genai_errors.ServerError(code, {"message": message, "status": "X"})


class TestRetryableConsistency:
    def test_retryable_types_exact(self):
        assert RETRYABLE_TYPES == {
            TriageErrorType.RATE_LIMIT,
            TriageErrorType.SERVER_ERROR,
            TriageErrorType.CONNECTION,
            TriageErrorType.TIMEOUT,
        }


class TestClassifyClientError:
    def test_429_rate_limit(self):
        info = classify(_client_error(429))
        assert info.type is TriageErrorType.RATE_LIMIT
        assert info.retryable is True

    @pytest.mark.parametrize("code", [401, 403])
    def test_auth(self, code):
        info = classify(_client_error(code))
        assert info.type is TriageErrorType.AUTHENTICATION
        assert info.retryable is False

    def test_408_timeout(self):
        info = classify(_client_error(408))
        assert info.type is TriageErrorType.TIMEOUT
        assert info.retryable is True

    @pytest.mark.parametrize("code", [400, 404, 413, 422])
    def test_other_4xx_bad_request(self, code):
        info = classify(_client_error(code))
        assert info.type is TriageErrorType.BAD_REQUEST
        assert info.retryable is False

    def test_message_extracted(self):
        info = classify(_client_error(429, message="slow down"))
        assert "slow down" in info.message


class TestClassifyServerError:
    @pytest.mark.parametrize("code", [500, 503])
    def test_server_error(self, code):
        info = classify(_server_error(code))
        assert info.type is TriageErrorType.SERVER_ERROR
        assert info.retryable is True


class TestClassifyNetwork:
    def test_timeout_before_transport(self):
        # httpx.ConnectTimeout is both a TimeoutException and a TransportError -
        # must resolve to TIMEOUT, proving ordering.
        info = classify(httpx.ConnectTimeout("slow"))
        assert info.type is TriageErrorType.TIMEOUT
        assert info.retryable is True

    def test_connect_error(self):
        info = classify(httpx.ConnectError("refused"))
        assert info.type is TriageErrorType.CONNECTION
        assert info.retryable is True


class TestClassifyValidation:
    def test_pydantic_validation_error(self):
        from models import TriageDecision

        try:
            TriageDecision(system="nope", severity="High", action="Restart", confidence=None)
        except ValidationError as exc:
            info = classify(exc)
        else:
            pytest.fail("expected ValidationError")
        assert info.type is TriageErrorType.SCHEMA_VALIDATION
        assert info.retryable is False


class TestClassifyFallback:
    def test_generic_exception(self):
        info = classify(ValueError("boom"))
        assert info.type is TriageErrorType.UNKNOWN
        assert info.retryable is False
        assert "boom" in info.message

    def test_never_raises_on_weird_input(self):
        class Weird(Exception):
            pass

        info = classify(Weird())
        assert info.type is TriageErrorType.UNKNOWN


class TestRetryableMatchesTable:
    @pytest.mark.parametrize(
        "exc",
        [
            _client_error(429),
            _client_error(401),
            _client_error(408),
            _client_error(400),
            _server_error(500),
            httpx.ConnectTimeout("slow"),
            httpx.ConnectError("refused"),
            ValueError("x"),
        ],
    )
    def test_consistency(self, exc):
        info = classify(exc)
        assert info.retryable == (info.type in RETRYABLE_TYPES)


class TestFromBlocked:
    def test_block_reason_wins(self):
        info = from_blocked(finish_reason="SAFETY", block_reason="PROHIBITED_CONTENT")
        assert info.type is TriageErrorType.REFUSAL
        assert info.retryable is False
        assert "PROHIBITED_CONTENT" in info.message

    def test_finish_reason_only(self):
        info = from_blocked(finish_reason="RECITATION")
        assert "RECITATION" in info.message

    def test_neither_given_still_valid_message(self):
        info = from_blocked(finish_reason=None)
        assert len(info.message) > 0
