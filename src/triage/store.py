"""Persistence: TriageResult and BatchJob rows via SQLModel.

Nested domain objects (decision/error/usage) are stored as JSON text columns
and reconstructed on read - keeps the schema stable while models.py evolves.
"""
from __future__ import annotations

from datetime import datetime

from models import (
    BatchJob,
    BatchJobStatus,
    ErrorInfo,
    TokenUsage,
    TriageDecision,
    TriageResult,
)
from sqlmodel import Field as SQLField
from sqlmodel import Session, SQLModel, create_engine, select

__all__ = ["BatchJobRow", "Store", "TriageResultRow"]


class TriageResultRow(SQLModel, table=True):
    __tablename__ = "triage_results"

    incident_id: str = SQLField(primary_key=True)
    complaint: str
    status: str
    decision_json: str | None = None
    error_json: str | None = None
    raw_response: str | None = None
    usage_json: str
    model: str
    attempts: int
    latency_ms: int
    created_at: str  # ISO-8601, stored explicitly to preserve tzinfo (SQLite drops it on DateTime columns)


class BatchJobRow(SQLModel, table=True):
    __tablename__ = "batch_jobs"

    job_id: str = SQLField(primary_key=True)
    provider_batch_id: str | None = None
    status: str
    total: int
    succeeded: int
    failed: int
    created_at: str  # ISO-8601, see TriageResultRow.created_at
    completed_at: str | None = None


def _result_to_row(result: TriageResult) -> TriageResultRow:
    return TriageResultRow(
        incident_id=result.incident_id,
        complaint=result.complaint,
        status=result.status,
        decision_json=result.decision.model_dump_json() if result.decision else None,
        error_json=result.error.model_dump_json() if result.error else None,
        raw_response=result.raw_response,
        usage_json=result.usage.model_dump_json(),
        model=result.model,
        attempts=result.attempts,
        latency_ms=result.latency_ms,
        created_at=result.created_at.isoformat(),
    )


def _row_to_result(row: TriageResultRow) -> TriageResult:
    return TriageResult(
        incident_id=row.incident_id,
        complaint=row.complaint,
        status=row.status,
        decision=TriageDecision.model_validate_json(row.decision_json) if row.decision_json else None,
        error=ErrorInfo.model_validate_json(row.error_json) if row.error_json else None,
        raw_response=row.raw_response,
        usage=TokenUsage.model_validate_json(row.usage_json),
        model=row.model,
        attempts=row.attempts,
        latency_ms=row.latency_ms,
        created_at=datetime.fromisoformat(row.created_at),
    )


def _job_to_row(job: BatchJob) -> BatchJobRow:
    return BatchJobRow(
        job_id=job.job_id,
        provider_batch_id=job.provider_batch_id,
        status=job.status.value,
        total=job.total,
        succeeded=job.succeeded,
        failed=job.failed,
        created_at=job.created_at.isoformat(),
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
    )


def _row_to_job(row: BatchJobRow) -> BatchJob:
    return BatchJob(
        job_id=row.job_id,
        provider_batch_id=row.provider_batch_id,
        status=BatchJobStatus(row.status),
        total=row.total,
        succeeded=row.succeeded,
        failed=row.failed,
        created_at=datetime.fromisoformat(row.created_at),
        completed_at=datetime.fromisoformat(row.completed_at) if row.completed_at else None,
    )


class Store:
    def __init__(self, database_url: str) -> None:
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self._engine = create_engine(database_url, connect_args=connect_args)
        SQLModel.metadata.create_all(self._engine)

    def save_result(self, result: TriageResult) -> None:
        row = _result_to_row(result)
        with Session(self._engine) as session:
            session.merge(row)  # upsert by primary key
            session.commit()

    def get_result(self, incident_id: str) -> TriageResult | None:
        with Session(self._engine) as session:
            row = session.get(TriageResultRow, incident_id)
        return _row_to_result(row) if row is not None else None

    def list_results(self, limit: int = 100) -> list[TriageResult]:
        with Session(self._engine) as session:
            rows = session.exec(select(TriageResultRow).limit(limit)).all()
        return [_row_to_result(r) for r in rows]

    def save_batch_job(self, job: BatchJob) -> None:
        row = _job_to_row(job)
        with Session(self._engine) as session:
            session.merge(row)
            session.commit()

    def get_batch_job(self, job_id: str) -> BatchJob | None:
        with Session(self._engine) as session:
            row = session.get(BatchJobRow, job_id)
        return _row_to_job(row) if row is not None else None
