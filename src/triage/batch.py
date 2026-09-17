"""Batch orchestration: bounded-concurrency async fan-out over triage_async.

Gemini has no direct equivalent of Anthropic's 50%-off Batches API at the
time of writing, so the batch path is concurrent live requests capped by
`settings.batch_max_concurrency` rather than an async job submitted to the
provider. Revisit if/when Gemini ships a comparable batch endpoint.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from config import get_settings
from models import BatchJob, BatchJobStatus, TriageRequest
from service import triage_async
from store import Store

__all__ = ["run_batch"]


async def run_batch(requests: list[TriageRequest], store: Store, job_id: str) -> BatchJob:
    """Run every request with bounded concurrency, persisting each result as
    it completes, and persisting the job's terminal state at the end.

    Assumes a BatchJob row with this job_id already exists (status=pending),
    created by the caller (e.g. the API endpoint) before this runs.
    """
    settings = get_settings()
    total = len(requests)

    running = _reload_job(store, job_id, total)
    running = BatchJob(
        job_id=running.job_id,
        provider_batch_id=running.provider_batch_id,
        status=BatchJobStatus.RUNNING,
        total=total,
        succeeded=0,
        failed=0,
        created_at=running.created_at,
        completed_at=None,
    )
    store.save_batch_job(running)

    semaphore = asyncio.Semaphore(settings.batch_max_concurrency)
    succeeded = 0
    failed = 0

    async def _run_one(request: TriageRequest) -> None:
        nonlocal succeeded, failed
        async with semaphore:
            result = await triage_async(request)
            store.save_result(result)
            if result.status == "success":
                succeeded += 1
            else:
                failed += 1

    if requests:
        await asyncio.gather(*(_run_one(r) for r in requests))

    completed = BatchJob(
        job_id=running.job_id,
        provider_batch_id=running.provider_batch_id,
        status=BatchJobStatus.COMPLETED,
        total=total,
        succeeded=succeeded,
        failed=failed,
        created_at=running.created_at,
        completed_at=datetime.now(timezone.utc),
    )
    store.save_batch_job(completed)
    return completed


def _reload_job(store: Store, job_id: str, total: int) -> BatchJob:
    job = store.get_batch_job(job_id)
    if job is not None:
        return job
    # caller didn't pre-create the row; create a pending one now
    job = BatchJob(job_id=job_id, status=BatchJobStatus.PENDING, total=total)
    store.save_batch_job(job)
    return job
