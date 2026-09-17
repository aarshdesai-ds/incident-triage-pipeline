"""Reproduce the original notebook's 55-incident batch run against the new
pipeline: bounded concurrency, structured output, persisted results, and a
JSON export equivalent to the notebook's `triage_results.json`.

Usage (from src/triage/, with GEMINI_API_KEY set):
    python run_sample_batch.py
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid

from batch import run_batch
from config import get_settings
from models import TriageRequest
from store import Store

SAMPLE_COMPLAINTS = [
    # Database
    "Our database connection pool is exhausted! 500 errors across the board.",
    "Queries that used to take 200ms are now taking 8 seconds.",
    "Getting intermittent 'connection refused' errors from the primary DB.",
    "Nightly backup job failed silently for the third night in a row.",
    "Replication lag between primary and replica is now over 10 minutes.",
    "We hit a deadlock during checkout and lost several orders.",
    "Disk usage on the DB server just crossed 95%.",
    "A migration script partially ran and now two tables are out of sync.",
    "Read replica is serving stale data to the reporting dashboard.",
    "Users are seeing duplicate rows after the last schema change.",
    "Connection pool warnings started showing up in the logs an hour ago.",
    "The staging database seems fine but production queries are timing out.",
    # Frontend
    "The login page loads, but clicking the Sign In button does nothing.",
    "Checkout page shows a blank white screen after adding a payment method.",
    "Mobile users report the nav menu overlaps the page content.",
    "Images on the product page aren't loading, just broken icons.",
    "Form validation lets users submit an empty required field.",
    "The dashboard chart never finishes loading, spinner just spins forever.",
    "CSS seems broken after the last deploy, layout is completely off.",
    "Users can't scroll past the third section on the pricing page.",
    "The 'Forgot Password' link goes to a 404 page.",
    "Search results page shows results from yesterday's cache, not live data.",
    "Users report the site looks fine on Chrome but is unusable on Safari.",
    "A JS error in the console is blocking the entire signup form from rendering.",
    "The dropdown menu flickers and closes before users can select an option.",
    # Network
    "DNS isn't resolving for our staging subdomain since this morning.",
    "We're seeing 40% packet loss between the app servers and the cache layer.",
    "SSL certificate on the API gateway expired an hour ago.",
    "VPN access for the remote team has been down since 9am.",
    "Load balancer is routing traffic unevenly, one instance is at 100% CPU.",
    "CDN is serving an old version of our JS bundle to some regions.",
    "Firewall rule change seems to be blocking traffic from our partner API.",
    "Latency to the EU region has tripled in the last 30 minutes.",
    "We're getting sporadic timeouts calling the third-party payment gateway.",
    "Internal service mesh is reporting intermittent connection drops.",
    "A misconfigured DNS record is pointing our subdomain to the wrong IP.",
    # Ambiguous / Unknown
    "Something feels off with the app today, can't quite pin it down.",
    "A few customers emailed saying 'the site is slow,' no other details.",
    "Support ticket just says 'nothing works,' no screenshots or logs.",
    "Intermittent issue that we can't reproduce on our end.",
    "One user reports data missing from their account, unclear which system.",
    "Multiple small glitches reported today, unclear if related.",
    "App feels sluggish across the board but no errors in the logs.",
    "A vague complaint about 'weird behavior' after the last release.",
    # High severity / urgent
    "Entire site is returning 502 errors, complete outage.",
    "Payment processing is down, customers can't complete purchases.",
    "We just got paged: production is throwing errors at 90% of requests.",
    "Data appears to have been overwritten for a subset of user accounts.",
    "Security team flagged unusual traffic patterns that look like a DDoS.",
    # Low severity / minor
    "A tooltip is slightly misaligned on the settings page.",
    "Footer copyright year is still showing last year.",
    "Minor typo in the confirmation email subject line.",
    "One icon is using the wrong color in dark mode.",
    "A non-critical background job is running a few minutes late.",
    "Placeholder text is showing in a field that should be empty by default.",
]


async def _run() -> list[dict]:
    settings = get_settings()
    store = Store(settings.database_url)
    requests = [
        TriageRequest(incident_id=f"sample-{i:02d}", complaint=text)
        for i, text in enumerate(SAMPLE_COMPLAINTS, start=1)
    ]

    job_id = str(uuid.uuid4())
    print(f"Running {len(requests)} incidents (job {job_id}, model={settings.model}, "
          f"concurrency={settings.batch_max_concurrency})...")

    job = await run_batch(requests, store, job_id=job_id)

    results = [store.get_result(r.incident_id) for r in requests]
    export = [
        {
            "id": r.incident_id,
            "complaint": r.complaint,
            "status": r.status,
            "decision": r.decision.model_dump(mode="json") if r.decision else None,
            "error": r.error.model_dump(mode="json") if r.error else None,
            "raw_response": r.raw_response,
            "tokens": r.usage.total_tokens,
        }
        for r in results
        if r is not None
    ]

    print(f"Done: {job.succeeded}/{job.total} succeeded, {job.failed} failed.")
    return export


if __name__ == "__main__":
    try:
        export = asyncio.run(_run())
    except KeyboardInterrupt:
        sys.exit(1)

    with open("triage_results.json", "w", encoding="utf-8") as f:
        json.dump(export, f, indent=2)
    print("Wrote triage_results.json")
