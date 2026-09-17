"""Unit tests for models.py.

Run:  pytest test_models.py -v
Once packaged, change the import to `from triage.models import ...`.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import pytest
from models import (
    Action,
    BatchJob,
    BatchJobStatus,
    ErrorInfo,
    Severity,
    System,
    TokenUsage,
    TriageDecision,
    TriageErrorType,
    TriageRequest,
    TriageResult,
)
from pydantic import ValidationError


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
@pytest.fixture
def decision() -> TriageDecision:
    return TriageDecision(
        system="Frontend", severity="High", action="Investigate", confidence=0.9
    )


@pytest.fixture
def usage() -> TokenUsage:
    return TokenUsage(input_tokens=100, output_tokens=53)


@pytest.fixture
def error_info() -> ErrorInfo:
    return ErrorInfo(type="rate_limit", message="429 slow down", retryable=True)


def _success_result(**overrides) -> TriageResult:
    base = {
        "incident_id": "INC-1",
        "complaint": "login button dead",
        "status": "success",
        "decision": TriageDecision(
            system="Frontend", severity="High", action="Investigate", confidence=0.9
        ),
        "usage": TokenUsage(input_tokens=100, output_tokens=53),
        "model": "claude-haiku-4-5",
        "attempts": 1,
        "latency_ms": 42,
    }
    base.update(overrides)
    return TriageResult(**base)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class TestEnums:
    def test_values_are_exact_titlecase(self):
        assert [s.value for s in System] == ["Database", "Frontend", "Network", "Unknown"]
        assert [s.value for s in Severity] == ["High", "Medium", "Low"]
        assert [a.value for a in Action] == ["Restart", "Scale", "Patch", "Investigate"]

    def test_str_backed(self):
        assert System.DATABASE == "Database"
        assert isinstance(System.DATABASE, str)

    def test_no_investigate_typo(self):
        # regression: was once "Inverstigate"
        assert Action.INVESTIGATE.value == "Investigate"


# --------------------------------------------------------------------------- #
# TriageDecision  (the wire model — what Claude must return)
# --------------------------------------------------------------------------- #
class TestTriageDecision:
    def test_valid(self, decision):
        assert decision.system is System.FRONTEND
        assert decision.confidence == 0.9

    def test_confidence_may_be_null_but_key_required(self):
        assert TriageDecision(
            system="Database", severity="Low", action="Restart", confidence=None
        ).confidence is None
        with pytest.raises(ValidationError):
            TriageDecision(system="Database", severity="Low", action="Restart")

    @pytest.mark.parametrize("bad", ["frontend", "FRONTEND", "Front End", "", "Api"])
    def test_rejects_bad_system(self, bad):
        with pytest.raises(ValidationError):
            TriageDecision(system=bad, severity="High", action="Restart", confidence=None)

    @pytest.mark.parametrize("bad", [-0.01, 1.01, 2, -5])
    def test_confidence_bounds(self, bad):
        with pytest.raises(ValidationError):
            TriageDecision(
                system="Network", severity="Medium", action="Scale", confidence=bad
            )

    def test_extra_key_forbidden(self):
        with pytest.raises(ValidationError):
            TriageDecision(
                system="Network", severity="Medium", action="Scale",
                confidence=None, reasoning="because",
            )

    def test_frozen(self, decision):
        with pytest.raises(ValidationError):
            decision.severity = Severity.LOW

    def test_json_schema_is_strict_clean(self):
        schema = TriageDecision.model_json_schema()
        assert schema["additionalProperties"] is False
        # every property must be required for strict structured outputs
        assert set(schema["required"]) == set(schema["properties"].keys())
        assert set(schema["required"]) == {"system", "severity", "action", "confidence"}


# --------------------------------------------------------------------------- #
# TriageRequest  (validated input)
# --------------------------------------------------------------------------- #
class TestTriageRequest:
    def test_strips_whitespace(self):
        r = TriageRequest(incident_id="  INC-9  ", complaint="   db down   ")
        assert r.incident_id == "INC-9"
        assert r.complaint == "db down"

    @pytest.mark.parametrize("blank", ["", "   ", "\t", "\n  \n"])
    def test_blank_complaint_rejected(self, blank):
        with pytest.raises(ValidationError):
            TriageRequest(incident_id="x", complaint=blank)

    @pytest.mark.parametrize("blank", ["", "   ", "\t"])
    def test_blank_incident_id_rejected(self, blank):
        with pytest.raises(ValidationError):
            TriageRequest(incident_id=blank, complaint="ok")

    def test_max_length(self):
        TriageRequest(incident_id="x", complaint="a" * 4000)
        with pytest.raises(ValidationError):
            TriageRequest(incident_id="x", complaint="a" * 4001)

    def test_optional_fields_default_none(self):
        r = TriageRequest(incident_id="x", complaint="y")
        assert r.source is None and r.received_at is None

    def test_naive_datetime_rejected(self):
        with pytest.raises(ValidationError):
            TriageRequest(incident_id="x", complaint="y", received_at=datetime(2026, 1, 1))  # noqa: DTZ001 - deliberately naive, that's what's under test

    def test_aware_datetime_accepted(self):
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert TriageRequest(incident_id="x", complaint="y", received_at=ts).received_at == ts

    def test_extra_key_forbidden(self):
        with pytest.raises(ValidationError):
            TriageRequest(incident_id="x", complaint="y", priority="P1")


# --------------------------------------------------------------------------- #
# TokenUsage
# --------------------------------------------------------------------------- #
class TestTokenUsage:
    def test_total(self, usage):
        assert usage.total_tokens == 153  # matches notebook smoke test

    def test_cache_reads_not_in_total(self):
        u = TokenUsage(input_tokens=10, output_tokens=5, cache_read_input_tokens=999)
        assert u.total_tokens == 15

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"input_tokens": -1, "output_tokens": 0},
            {"input_tokens": 0, "output_tokens": -1},
            {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": -1},
        ],
    )
    def test_negative_rejected(self, kwargs):
        with pytest.raises(ValidationError):
            TokenUsage(**kwargs)

    def test_frozen(self, usage):
        with pytest.raises(ValidationError):
            usage.input_tokens = 0

    def test_total_tokens_serialized(self, usage):
        # computed_field must appear in the dump
        assert json.loads(usage.model_dump_json())["total_tokens"] == 153


# --------------------------------------------------------------------------- #
# ErrorInfo
# --------------------------------------------------------------------------- #
class TestErrorInfo:
    def test_valid(self, error_info):
        assert error_info.type is TriageErrorType.RATE_LIMIT
        assert error_info.request_id is None

    def test_message_required_nonblank(self):
        with pytest.raises(ValidationError):
            ErrorInfo(type="unknown", message="", retryable=False)

    def test_unknown_error_type_rejected(self):
        with pytest.raises(ValidationError):
            ErrorInfo(type="teapot", message="x", retryable=False)

    def test_frozen(self, error_info):
        with pytest.raises(ValidationError):
            error_info.retryable = False


# --------------------------------------------------------------------------- #
# TriageResult  (envelope + status invariant)
# --------------------------------------------------------------------------- #
class TestTriageResult:
    def test_success_ok(self):
        r = _success_result()
        assert r.status == "success" and r.decision is not None and r.error is None

    def test_error_ok(self, error_info):
        r = TriageResult(
            incident_id="x", complaint="y", status="error", error=error_info,
            usage=TokenUsage(input_tokens=0, output_tokens=0),
            model="claude-haiku-4-5", attempts=3, latency_ms=5,
        )
        assert r.decision is None

    def test_success_requires_decision(self):
        with pytest.raises(ValidationError):
            _success_result(decision=None)

    def test_success_forbids_error(self, error_info):
        with pytest.raises(ValidationError):
            _success_result(error=error_info)

    def test_error_requires_error(self):
        with pytest.raises(ValidationError):
            TriageResult(
                incident_id="x", complaint="y", status="error",
                usage=TokenUsage(input_tokens=0, output_tokens=0),
                model="m", attempts=1, latency_ms=1,
            )

    def test_error_forbids_decision(self, decision, error_info):
        with pytest.raises(ValidationError):
            TriageResult(
                incident_id="x", complaint="y", status="error",
                decision=decision, error=error_info,
                usage=TokenUsage(input_tokens=0, output_tokens=0),
                model="m", attempts=1, latency_ms=1,
            )

    @pytest.mark.parametrize("status", ["Success", "SUCCESS", "ok", "failed", ""])
    def test_status_literal_enforced(self, status):
        with pytest.raises(ValidationError):
            _success_result(status=status)

    @pytest.mark.parametrize("kwargs", [{"attempts": 0}, {"attempts": -1}, {"latency_ms": -1}])
    def test_numeric_bounds(self, kwargs):
        with pytest.raises(ValidationError):
            _success_result(**kwargs)

    def test_created_at_is_tz_aware_utc(self):
        r = _success_result()
        assert r.created_at.tzinfo is not None
        assert r.created_at.utcoffset().total_seconds() == 0

    def test_json_round_trip(self):
        r = _success_result()
        assert TriageResult.model_validate(json.loads(r.model_dump_json())) == r

    def test_extra_key_forbidden(self):
        with pytest.raises(ValidationError):
            _success_result(owner="team-frontend")


# --------------------------------------------------------------------------- #
# BatchJob
# --------------------------------------------------------------------------- #
class TestBatchJob:
    def test_running_ok(self):
        j = BatchJob(job_id="j1", status="running", total=10, succeeded=3, failed=1)
        assert j.provider_batch_id is None and j.completed_at is None

    def test_counts_cannot_exceed_total(self):
        with pytest.raises(ValidationError):
            BatchJob(job_id="j", status="running", total=2, succeeded=2, failed=1)

    def test_completed_requires_completed_at(self):
        with pytest.raises(ValidationError):
            BatchJob(job_id="j", status="completed", total=5, succeeded=5)

    def test_completed_requires_full_accounting(self):
        with pytest.raises(ValidationError):
            BatchJob(
                job_id="j", status="completed", total=5, succeeded=2, failed=1,
                completed_at=datetime.now(timezone.utc),
            )

    def test_completed_ok(self):
        j = BatchJob(
            job_id="j", status="completed", total=5, succeeded=4, failed=1,
            completed_at=datetime.now(timezone.utc),
        )
        assert j.status is BatchJobStatus.COMPLETED

    def test_negative_counts_rejected(self):
        with pytest.raises(ValidationError):
            BatchJob(job_id="j", status="pending", total=-1)


# --------------------------------------------------------------------------- #
# Module hygiene
# --------------------------------------------------------------------------- #
class TestModuleHygiene:
    def test_all_exports_resolve(self):
        import models
        for name in models.__all__:
            assert hasattr(models, name), f"__all__ names missing symbol: {name}"

    def test_no_pydantic_warnings_on_import(self):
        import importlib

        import models
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            importlib.reload(models)
