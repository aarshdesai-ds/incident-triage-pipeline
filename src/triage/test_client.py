"""Unit tests for client.py. Gemini calls are mocked - no network / API key needed.

Run: pytest src/triage/test_client.py -v
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import client as client_mod
import pytest
from client import RawTriageResponse, call_gemini, call_gemini_async
from config import get_settings
from models import Action, Severity, System, TriageDecision, TriageErrorType


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    get_settings.cache_clear()
    client_mod.reset_client()
    yield
    get_settings.cache_clear()
    client_mod.reset_client()


def _fake_response(
    decision: TriageDecision | None = None,
    text: str | None = None,
    finish_reason: str | None = "STOP",
    block_reason: str | None = None,
    prompt_tokens: int = 40,
    output_tokens: int = 10,
):
    candidate = SimpleNamespace(finish_reason=finish_reason)
    prompt_feedback = SimpleNamespace(block_reason=block_reason) if block_reason else None
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=output_tokens,
        cached_content_token_count=0,
    )
    resp = SimpleNamespace(
        parsed=decision,
        text=text,
        candidates=[candidate] if finish_reason is not None else [],
        prompt_feedback=prompt_feedback,
        usage_metadata=usage,
    )
    return resp


def _patch_generate_content(monkeypatch, response=None, side_effect=None):
    fake_models = MagicMock()
    if side_effect is not None:
        fake_models.generate_content.side_effect = side_effect
    else:
        fake_models.generate_content.return_value = response
    fake_client = SimpleNamespace(models=fake_models)
    monkeypatch.setattr(client_mod, "get_client", lambda: fake_client)
    return fake_models


class TestCallGeminiSuccess:
    def test_returns_decision(self, monkeypatch):
        decision = TriageDecision(system=System.DATABASE, severity=Severity.HIGH, action=Action.RESTART, confidence=0.9)
        response = _fake_response(decision=decision, text='{"system":"Database", ...}')
        _patch_generate_content(monkeypatch, response=response)

        raw = call_gemini("db is down")
        assert isinstance(raw, RawTriageResponse)
        assert raw.decision == decision
        assert raw.error is None
        assert raw.usage.total_tokens == 50

    def test_falls_back_to_manual_parse_when_not_parsed(self, monkeypatch):
        # .parsed is None but .text is valid JSON -> manual parse succeeds
        text = '{"system":"Frontend","severity":"Low","action":"Investigate","confidence":null}'
        response = _fake_response(decision=None, text=text)
        _patch_generate_content(monkeypatch, response=response)

        raw = call_gemini("minor css glitch")
        assert raw.error is None
        assert raw.decision.system is System.FRONTEND
        assert raw.decision.confidence is None

    def test_dict_parsed_is_validated_into_triage_decision(self, monkeypatch):
        # the real Gemini shape: .parsed is a plain dict, since we hand the
        # SDK a types.Schema rather than the TriageDecision class directly.
        parsed_dict = {"system": "Network", "severity": "Medium", "action": "Scale", "confidence": 0.7}
        response = _fake_response(decision=parsed_dict, text='{"system": "Network", ...}')
        _patch_generate_content(monkeypatch, response=response)

        raw = call_gemini("load balancer overloaded")
        assert raw.error is None
        assert raw.decision == TriageDecision(system=System.NETWORK, severity=Severity.MEDIUM, action=Action.SCALE, confidence=0.7)

    def test_invalid_dict_parsed_becomes_schema_validation_error(self, monkeypatch):
        parsed_dict = {"system": "not-a-system", "severity": "High", "action": "Restart", "confidence": None}
        response = _fake_response(decision=parsed_dict, text="irrelevant")
        _patch_generate_content(monkeypatch, response=response)

        raw = call_gemini("weird")
        assert raw.decision is None
        assert raw.error.type is TriageErrorType.SCHEMA_VALIDATION


class TestCallGeminiBlocked:
    def test_blocked_by_prompt(self, monkeypatch):
        response = _fake_response(decision=None, text=None, finish_reason=None, block_reason="SAFETY")
        _patch_generate_content(monkeypatch, response=response)

        raw = call_gemini("something bad")
        assert raw.decision is None
        assert raw.error is not None
        assert raw.error.type is TriageErrorType.REFUSAL
        assert "SAFETY" in raw.error.message

    def test_blocked_by_finish_reason(self, monkeypatch):
        response = _fake_response(decision=None, text=None, finish_reason="RECITATION")
        _patch_generate_content(monkeypatch, response=response)

        raw = call_gemini("quote a poem")
        assert raw.error.type is TriageErrorType.REFUSAL
        assert "RECITATION" in raw.error.message

    def test_unparseable_text_becomes_schema_validation_error(self, monkeypatch):
        response = _fake_response(decision=None, text="not json at all")
        _patch_generate_content(monkeypatch, response=response)

        raw = call_gemini("weird output")
        assert raw.decision is None
        assert raw.error.type is TriageErrorType.SCHEMA_VALIDATION


class TestCallGeminiUsageWithNoMetadata:
    def test_missing_usage_metadata_defaults_zero(self, monkeypatch):
        decision = TriageDecision(system=System.UNKNOWN, severity=Severity.LOW, action=Action.INVESTIGATE, confidence=None)
        response = _fake_response(decision=decision)
        response.usage_metadata = None
        _patch_generate_content(monkeypatch, response=response)

        raw = call_gemini("x")
        assert raw.usage.total_tokens == 0


class TestCallGeminiPropagatesExceptions:
    def test_sdk_exception_propagates(self, monkeypatch):
        _patch_generate_content(monkeypatch, side_effect=RuntimeError("network blew up"))
        with pytest.raises(RuntimeError, match="network blew up"):
            call_gemini("x")


@pytest.mark.asyncio
class TestCallGeminiAsync:
    async def test_returns_decision(self, monkeypatch):
        decision = TriageDecision(system=System.NETWORK, severity=Severity.MEDIUM, action=Action.SCALE, confidence=0.5)
        response = _fake_response(decision=decision)

        async def _gen(*args, **kwargs):
            return response

        fake_models = SimpleNamespace(generate_content=_gen)
        fake_client = SimpleNamespace(aio=SimpleNamespace(models=fake_models))
        monkeypatch.setattr(client_mod, "get_client", lambda: fake_client)

        raw = await call_gemini_async("load balancer overloaded")
        assert raw.decision == decision
