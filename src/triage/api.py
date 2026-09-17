"""FastAPI service: single triage, batch submission, batch/result lookup."""
from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from batch import run_batch
from config import get_settings
from fastapi import BackgroundTasks, FastAPI, HTTPException
from models import BatchJob, BatchJobStatus, TriageRequest, TriageResult
from service import triage_async
from store import Store

__all__ = ["app"]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.store = Store(settings.database_url)
    yield


app = FastAPI(title="Incident Triage", lifespan=lifespan)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/triage", response_model=TriageResult)
async def triage_endpoint(request: TriageRequest) -> TriageResult:
    result = await triage_async(request)
    app.state.store.save_result(result)
    return result


@app.get("/triage/{incident_id}", response_model=TriageResult)
def get_triage(incident_id: str) -> TriageResult:
    result = app.state.store.get_result(incident_id)
    if result is None:
        raise HTTPException(status_code=404, detail="incident not found")
    return result


@app.post("/triage/batch", response_model=BatchJob, status_code=202)
async def submit_batch(requests: list[TriageRequest], background_tasks: BackgroundTasks) -> BatchJob:
    if not requests:
        raise HTTPException(status_code=400, detail="requests must not be empty")

    job = BatchJob(job_id=str(uuid.uuid4()), status=BatchJobStatus.PENDING, total=len(requests))
    app.state.store.save_batch_job(job)
    background_tasks.add_task(run_batch, requests, app.state.store, job.job_id)
    return job


@app.get("/triage/batch/{job_id}", response_model=BatchJob)
def get_batch(job_id: str) -> BatchJob:
    job = app.state.store.get_batch_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="batch job not found")
    return job
