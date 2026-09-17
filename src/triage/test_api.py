"""Unit tests for api.py via FastAPI's TestClient. service.triage_async and
batch.run_batch are mocked - no network / API key needed.
Run: pytest src/triage/test_api.py -v
"""
from __future__ import annotations

import api as api_mod
import pytest
from config import get_settings
from fastapi.testclient import TestClient
from models import (
    Action,
    Severity,
    System,
    TokenUsage,
    TriageDecision,
    TriageResult,
)


@pytest.fixture(autouse=True)
def _reset_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("TRIAGE_DATABASE_URL", f"sqlite:///{tmp_path / 'api.db'}")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    with TestClient(api_mod.app) as c:
        yield c


def _success(incident_id: str, complaint: str) -> TriageResult:
    return TriageResult(
        incident_id=incident_id, complaint=complaint, status="success",
        decision=TriageDecision(system=System.NETWORK, severity=Severity.MEDIUM, action=Action.SCALE, confidence=0.6),
        usage=TokenUsage(input_tokens=10, output_tokens=5), model="gemini-2.5-flash", attempts=1, latency_ms=50,
    )


class TestHealthz:
    def test_ok(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestTriageEndpoint:
    def test_success(self, client, monkeypatch):
        async def fake_triage_async(request):
            return _success(request.incident_id, request.complaint)

        monkeypatch.setattr(api_mod, "triage_async", fake_triage_async)

        resp = client.post("/triage", json={"incident_id": "INC-1", "complaint": "vpn is down"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "success"
        assert body["decision"]["system"] == "Network"

    def test_invalid_request_422(self, client):
        resp = client.post("/triage", json={"incident_id": "INC-1", "complaint": ""})
        assert resp.status_code == 422

    def test_result_persisted_and_retrievable(self, client, monkeypatch):
        async def fake_triage_async(request):
            return _success(request.incident_id, request.complaint)

        monkeypatch.setattr(api_mod, "triage_async", fake_triage_async)
        client.post("/triage", json={"incident_id": "INC-9", "complaint": "db slow"})

        resp = client.get("/triage/INC-9")
        assert resp.status_code == 200
        assert resp.json()["incident_id"] == "INC-9"

    def test_unknown_incident_404(self, client):
        resp = client.get("/triage/does-not-exist")
        assert resp.status_code == 404


class TestBatchEndpoint:
    def test_submit_returns_pending_job(self, client, monkeypatch):
        async def fake_run_batch(requests, store, job_id):
            return None  # background task; return value unused here

        monkeypatch.setattr(api_mod, "run_batch", fake_run_batch)

        resp = client.post(
            "/triage/batch",
            json=[{"incident_id": "A", "complaint": "x"}, {"incident_id": "B", "complaint": "y"}],
        )
        assert resp.status_code == 202
        body = resp.json()
        assert body["status"] == "pending"
        assert body["total"] == 2

        # job row was created synchronously, before the background task runs
        fetched = client.get(f"/triage/batch/{body['job_id']}")
        assert fetched.status_code == 200

    def test_empty_batch_rejected(self, client):
        resp = client.post("/triage/batch", json=[])
        assert resp.status_code == 400

    def test_unknown_job_404(self, client):
        resp = client.get("/triage/batch/does-not-exist")
        assert resp.status_code == 404
