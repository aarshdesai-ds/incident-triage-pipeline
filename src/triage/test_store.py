"""Unit tests for store.py against a real (temp-file) SQLite DB.
Run: pytest src/triage/test_store.py -v
"""
from __future__ import annotations

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
    TriageResult,
)
from store import Store


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    return Store(f"sqlite:///{db_path}")


def _success_result(incident_id: str = "INC-1") -> TriageResult:
    return TriageResult(
        incident_id=incident_id,
        complaint="db is down",
        status="success",
        decision=TriageDecision(system=System.DATABASE, severity=Severity.HIGH, action=Action.RESTART, confidence=0.9),
        raw_response='{"system":"Database"}',
        usage=TokenUsage(input_tokens=40, output_tokens=10),
        model="gemini-2.5-flash",
        attempts=1,
        latency_ms=120,
    )


def _error_result(incident_id: str = "INC-2") -> TriageResult:
    return TriageResult(
        incident_id=incident_id,
        complaint="weird thing",
        status="error",
        error=ErrorInfo(type=TriageErrorType.RATE_LIMIT, message="429", retryable=True),
        usage=TokenUsage(input_tokens=0, output_tokens=0),
        model="gemini-2.5-flash",
        attempts=3,
        latency_ms=5,
    )


class TestResultRoundTrip:
    def test_success_round_trip(self, store):
        original = _success_result()
        store.save_result(original)
        fetched = store.get_result("INC-1")
        assert fetched == original

    def test_error_round_trip(self, store):
        original = _error_result()
        store.save_result(original)
        fetched = store.get_result("INC-2")
        assert fetched == original

    def test_missing_returns_none(self, store):
        assert store.get_result("nope") is None

    def test_upsert_overwrites(self, store):
        store.save_result(_success_result("INC-3"))
        updated = _success_result("INC-3").model_copy(update={"attempts": 2})
        store.save_result(updated)
        fetched = store.get_result("INC-3")
        assert fetched.attempts == 2

    def test_list_results(self, store):
        store.save_result(_success_result("A"))
        store.save_result(_success_result("B"))
        results = store.list_results()
        assert {r.incident_id for r in results} == {"A", "B"}


class TestBatchJobRoundTrip:
    def test_pending_round_trip(self, store):
        job = BatchJob(job_id="job-1", status=BatchJobStatus.PENDING, total=5)
        store.save_batch_job(job)
        fetched = store.get_batch_job("job-1")
        assert fetched == job

    def test_completed_round_trip(self, store):
        job = BatchJob(
            job_id="job-2", status=BatchJobStatus.COMPLETED, total=3,
            succeeded=2, failed=1, completed_at=datetime.now(timezone.utc),
        )
        store.save_batch_job(job)
        fetched = store.get_batch_job("job-2")
        assert fetched == job

    def test_missing_returns_none(self, store):
        assert store.get_batch_job("nope") is None

    def test_upsert_transitions_status(self, store):
        pending = BatchJob(job_id="job-3", status=BatchJobStatus.PENDING, total=2)
        store.save_batch_job(pending)
        running = BatchJob(job_id="job-3", status=BatchJobStatus.RUNNING, total=2, succeeded=1)
        store.save_batch_job(running)
        fetched = store.get_batch_job("job-3")
        assert fetched.status is BatchJobStatus.RUNNING
        assert fetched.succeeded == 1
