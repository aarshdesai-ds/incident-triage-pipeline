"""Unit tests for batch.py. service.triage_async is mocked.
Run: pytest src/triage/test_batch.py -v
"""
from __future__ import annotations

import batch as batch_mod
import pytest
from config import get_settings
from models import (
    Action,
    BatchJobStatus,
    ErrorInfo,
    Severity,
    System,
    TokenUsage,
    TriageDecision,
    TriageErrorType,
    TriageRequest,
)
from store import Store


@pytest.fixture(autouse=True)
def _reset_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path / 'batch.db'}")


def _requests(n: int) -> list[TriageRequest]:
    return [TriageRequest(incident_id=f"INC-{i}", complaint=f"issue {i}") for i in range(n)]


def _success(incident_id: str, complaint: str):
    from models import TriageResult

    return TriageResult(
        incident_id=incident_id, complaint=complaint, status="success",
        decision=TriageDecision(system=System.DATABASE, severity=Severity.HIGH, action=Action.RESTART, confidence=0.7),
        usage=TokenUsage(input_tokens=1, output_tokens=1), model="gemini-2.5-flash", attempts=1, latency_ms=1,
    )


def _failure(incident_id: str, complaint: str):
    from models import TriageResult

    return TriageResult(
        incident_id=incident_id, complaint=complaint, status="error",
        error=ErrorInfo(type=TriageErrorType.BAD_REQUEST, message="bad", retryable=False),
        usage=TokenUsage(input_tokens=0, output_tokens=0), model="gemini-2.5-flash", attempts=1, latency_ms=1,
    )


class TestRunBatch:
    @pytest.mark.asyncio
    async def test_all_succeed(self, monkeypatch, store):
        requests = _requests(5)

        async def fake_triage(request):
            return _success(request.incident_id, request.complaint)

        monkeypatch.setattr(batch_mod, "triage_async", fake_triage)

        job = await batch_mod.run_batch(requests, store, job_id="job-1")

        assert job.status is BatchJobStatus.COMPLETED
        assert job.total == 5
        assert job.succeeded == 5
        assert job.failed == 0
        assert job.completed_at is not None
        # every result persisted
        assert len(store.list_results()) == 5

    @pytest.mark.asyncio
    async def test_mixed_success_and_failure(self, monkeypatch, store):
        requests = _requests(4)

        async def fake_triage(request):
            n = int(request.incident_id.split("-")[1])
            return _success(request.incident_id, request.complaint) if n % 2 == 0 else _failure(request.incident_id, request.complaint)

        monkeypatch.setattr(batch_mod, "triage_async", fake_triage)
        job = await batch_mod.run_batch(requests, store, job_id="job-2")

        assert job.succeeded == 2
        assert job.failed == 2
        assert job.status is BatchJobStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_respects_concurrency_limit(self, monkeypatch, store):
        monkeypatch.setenv("TRIAGE_BATCH_MAX_CONCURRENCY", "2")
        get_settings.cache_clear()

        import asyncio

        in_flight = {"current": 0, "max_seen": 0}

        async def fake_triage(request):
            in_flight["current"] += 1
            in_flight["max_seen"] = max(in_flight["max_seen"], in_flight["current"])
            await asyncio.sleep(0.01)
            in_flight["current"] -= 1
            return _success(request.incident_id, request.complaint)

        monkeypatch.setattr(batch_mod, "triage_async", fake_triage)
        await batch_mod.run_batch(_requests(10), store, job_id="job-3")

        assert in_flight["max_seen"] <= 2

    @pytest.mark.asyncio
    async def test_empty_batch(self, monkeypatch, store):
        job = await batch_mod.run_batch([], store, job_id="job-4")
        assert job.status is BatchJobStatus.COMPLETED
        assert job.total == 0
        assert job.succeeded == 0
        assert job.failed == 0

    @pytest.mark.asyncio
    async def test_creates_job_row_if_missing(self, monkeypatch, store):
        async def fake_triage(request):
            return _success(request.incident_id, request.complaint)

        monkeypatch.setattr(batch_mod, "triage_async", fake_triage)
        # no pre-existing row for "job-5" - run_batch should create one
        job = await batch_mod.run_batch(_requests(2), store, job_id="job-5")
        assert store.get_batch_job("job-5") == job
