# Incident Triage Pipeline: Notebook to Production

A tested FastAPI service that turns messy, unstructured incident text into a structured `System` / `Severity` / `Action` triage decision, converted from a single-notebook Anthropic prototype into a provider-swapped, schema-validated, retried, persisted, and live-verified production pipeline on the Gemini API.

*This is a personal engineering project. The original notebook (`Sample_Incident_Triage_Pipeline_Anthropic_Final.ipynb`) used the Anthropic API; this rebuild runs on Google's Gemini API for cost reasons — see [Key Findings](#key-findings).*

---

## Table of Contents

- [Overview](#overview)
- [Design Questions](#design-questions)
- [Data Model](#data-model)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Ideas for Extension](#ideas-for-extension)

---

## Overview

The source notebook proved that an LLM could classify a cloud-incident complaint into a compact `System | Severity | Action` triage decision, with retry logic and batch execution bolted on around a single Anthropic API call. It worked, but it wasn't deployable: no input validation, no schema-enforced output (the model's `pipe | delimited | text` was never actually valid JSON, despite the notebook's own code trying to `json.loads()` it), a client rebuilt on every call, and a batch loop that paced itself with `time.sleep(1)`.

This project rebuilds that idea as a real service: Pydantic models enforce every boundary (request in, decision out, error shape, persisted record), a FastAPI app exposes single and batch triage endpoints, results are persisted to SQL, and the whole thing is unit-tested against mocks *and* verified against the live API before being called done.

---

## Design Questions

- Can an LLM's output be made genuinely schema-valid — not just prompted to look structured — so downstream code never has to defensively re-parse pipe-delimited text?
- Which failures are worth retrying (rate limits, 5xx, timeouts) versus which should fail fast (bad input, auth, a blocked/refused response)?
- Does a classification task this narrow need a "thinking" model at all, or does that just burn tokens for no benefit?
- What actually breaks when a notebook's happy-path assumptions meet a real, quota-limited, schema-picky provider API — and how much of that is discoverable without a live key?
- Is Anthropic's smallest model actually necessary here, or does a materially cheaper provider hold up on the same task?

---

## Data Model

Every request, decision, and error is a validated Pydantic model (`src/triage/models.py`) — nothing crosses a boundary as a loose dict.

### `TriageRequest` (input)

| Field | Type | Notes |
|---|---|---|
| `incident_id` | `str` | stripped, non-blank |
| `complaint` | `str` | 1–4000 chars, stripped, non-blank |
| `source` | `str \| None` | optional origin tag |
| `received_at` | `datetime \| None` | must be timezone-aware if given |

### `TriageDecision` (the model's structured output)

| Field | Type | Allowed values |
|---|---|---|
| `system` | enum | `Database`, `Frontend`, `Network`, `Unknown` |
| `severity` | enum | `High`, `Medium`, `Low` |
| `action` | enum | `Restart`, `Scale`, `Patch`, `Investigate` |
| `confidence` | `float \| None` | `0.0`–`1.0`, required key, nullable value |

### `TriageResult` (persisted / returned envelope)

| Field | Purpose |
|---|---|
| `status` | `"success"` or `"error"` — enforced mutually exclusive with `decision` / `error` |
| `decision` / `error` | exactly one is set, validated by a model-level invariant |
| `raw_response` | the model's raw text, kept for audit |
| `usage` | input/output/cached token counts |
| `attempts`, `latency_ms`, `model`, `created_at` | operational metadata |

`ErrorInfo` carries a machine-readable `type` (`rate_limit`, `bad_request`, `authentication`, `connection`, `server_error`, `schema_validation`, `refusal`, `timeout`, `unknown`) and a `retryable` flag, so the retry layer never has to inspect an exception message.

---

## Methodology

**Structured output, not prompted formatting.** The original notebook asked Claude for `System: X | Severity: Y | Action: Z` in prose and hoped for the best. This version sends Gemini a real schema (`response_mime_type="application/json"` + a `response_schema`) and validates whatever comes back through `TriageDecision` before it's allowed to exist as a decision.

**Error classification, not a single `except Exception`.** `errors.py` maps every Gemini/network exception to a `TriageErrorType` and a `retryable` flag via a most-specific-first `isinstance` chain, so `service.py`'s retry loop can make a real decision instead of blindly retrying everything three times like the notebook did.

**Retry orchestration separated from the API call.** `client.py` makes exactly one call and either returns a result or lets the exception propagate; `service.py` owns the retry loop, backoff, and result assembly. Blocked/refused responses (HTTP 200, empty content) are treated as a distinct, non-retryable case.

**Bounded-concurrency batch, not a sleep loop.** `batch.py` fans a batch out across `asyncio.gather` under a semaphore sized from config, replacing the notebook's serial `for` loop with a fixed `time.sleep(1)` between every call.

**Verify against the real API before calling it done.** Every unit test mocks the Gemini SDK — fast, free, deterministic — but mocks can't catch a provider rejecting your actual request shape. The full 55-incident sample batch was run against the live API specifically to surface exactly that class of bug (see below).

---

## Key Findings

**Provider switch (Anthropic → Gemini):**
- The rebuild targets `gemini-2.5-flash` instead of `claude-haiku-4-5` for cost — Anthropic's cheapest model still runs several times Gemini Flash's per-token rate for a task that's pure classification.
- `models.py` and `config.py` needed **zero changes** for the switch — only `errors.py` (exception taxonomy) and `client.py` (API call shape) are provider-specific, which is the whole point of keeping the domain model separate from the SDK layer.

**Bugs that only a live call could find** *(none of these are visible from reading either SDK's documentation alone)*:
- **Gemini's `response_schema` is a constrained OpenAPI-3.0 subset, not full JSON Schema** — it 400s on `additionalProperties`, which `TriageDecision.model_json_schema()` includes because of Pydantic's `extra="forbid"`. Fixed by hand-building the Gemini-facing schema directly from the `System`/`Severity`/`Action` enums instead of handing the SDK the Pydantic class.
- **Gemini 2.5 Flash's "thinking" is on by default and is billed out of `max_output_tokens`** — at a 256-token budget it silently consumed the entire thing and truncated the actual JSON answer to an unterminated fragment. Disabled outright (`thinking_budget=0`) since a four-field classification call has no real reasoning to do.
- **A credential that was configured but never loaded.** `Settings` deliberately never models the API key, but nothing called `load_dotenv()`, and the one path that did reference `.env` resolved it relative to the current working directory rather than a fixed location — so the key sat in the file, correctly formatted, and was never read. Every one of 55 requests failed with the same "no API key" error until this was found.
- **SQLite silently drops timezone info on datetime round-trip.** A `TriageResult` saved with a UTC-aware `created_at` came back naive after a `Store.save_result` → `Store.get_result` cycle. Fixed by storing timestamps as ISO-8601 strings under the app's own control instead of relying on the DB column type.

**Security:** a live, unrevoked Anthropic API key was found hardcoded in a file named `.env.example` (meant to be a committed template) during this project's development — a reminder that "example" files need the same scrutiny as real config before they're trusted.

---

## Project Structure

```
incident-triage/
│
├── README.md
├── Dockerfile
├── requirements.txt
├── pytest.ini
├── .gitignore
│
└── src/
    ├── .env.example                # template; real .env is git-ignored
    └── triage/
        ├── models.py                # Pydantic domain models (provider-agnostic)
        ├── config.py                # pydantic-settings; credential never modelled
        ├── errors.py                # Gemini exception -> ErrorInfo classification
        ├── client.py                # one Gemini call: schema, usage, blocked-response handling
        ├── service.py                # retry orchestration, sync + async
        ├── batch.py                  # bounded-concurrency fan-out
        ├── store.py                  # SQLModel persistence (SQLite by default)
        ├── api.py                    # FastAPI app
        ├── run_sample_batch.py       # reproduces the notebook's 55-incident run
        └── test_*.py                 # 149 tests, one file per module above
```

---

## Requirements

```
Python 3.11+
google-genai
pydantic
pydantic-settings
fastapi
uvicorn
sqlmodel
python-dotenv
httpx
```

Install dependencies:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/aarshdesai-ds/incident-triage-pipeline.git
   cd incident-triage-pipeline
   ```

2. Install dependencies (see [Requirements](#requirements)).

3. Add a Gemini API key:
   ```bash
   cp src/.env.example src/.env
   ```
   Edit `src/.env` and set `GEMINI_API_KEY` — get one at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).

4. Run the tests (no API key needed — every Gemini call is mocked):
   ```bash
   python -m pytest src/triage/ -v
   ```

5. Run the service:
   ```bash
   cd src/triage
   uvicorn api:app --reload
   ```
   - `POST /triage` — `{"incident_id": "...", "complaint": "..."}` → `TriageResult`
   - `GET /triage/{incident_id}` — fetch a stored result
   - `POST /triage/batch` — a list of the same request body → `202` with a `BatchJob`, runs in the background
   - `GET /triage/batch/{job_id}` — poll job status
   - `GET /healthz`

6. Or reproduce the notebook's original batch run against the new pipeline:
   ```bash
   cd src/triage
   python run_sample_batch.py
   ```
   Writes `triage_results.json` with real, schema-validated `decision` objects — not pipe-delimited text.

---

## Results Summary

**Live verification against the real Gemini API (2026-09-17):**

| Metric | Result |
|---|---|
| Incidents run | 55 / 55 |
| Succeeded | **55** |
| Failed | 0 |
| Avg. tokens / incident | ~116 |
| Unit tests passing | 149 / 149 |
| Ruff findings | 0 |

**Sample decisions from the live run:**

| Incident | System | Severity | Action | Confidence | Complaint |
|---|---|---|---|---|---|
| sample-01 | Database | High | Scale | 0.95 | "Our database connection pool is exhausted! 500 errors..." |
| sample-13 | Frontend | High | Investigate | 0.90 | "The login page loads, but clicking the Sign In button does nothing." |
| sample-45 | Frontend | High | Restart | 0.95 | "Entire site is returning 502 errors, complete outage." |
| sample-48 | Database | High | Investigate | 0.95 | "Data appears to have been overwritten for a subset of user accounts." |
| sample-52 | Frontend | Low | Patch | 0.80 | "Minor typo in the confirmation email subject line." |

Full detail for all 55 incidents is written to `triage_results.json` on each run of `run_sample_batch.py` (git-ignored — it's generated output, not source).

---

## Ideas for Extension

The current pipeline is a solid, verified foundation. Concrete next steps:

### 1. Build a Real Eval Set
The 55 sample incidents have no ground-truth labels — there's no way to measure classification *accuracy*, only that the pipeline runs end-to-end. Hand-labeling even 20–30 of them would turn `run_sample_batch.py` into a real regression gate for prompt or model changes.

### 2. Rate Limiting and Cost Guardrails
`config.py` already tracks token usage per result; there's no daily/per-key spend cap yet. A simple running-total check against a configured ceiling would prevent a runaway batch from an unbounded bill.

### 3. Structured Logging and Metrics
Right now, observability is "read the database." Structured JSON logs (per-request status, latency, token cost) plus a Prometheus counter/histogram would make this deployable behind a real dashboard.

### 4. Postgres in Production
`store.py` defaults to SQLite; the config already rejects SQLite in `TRIAGE_ENVIRONMENT=prod`, but there's no migration tooling (Alembic or similar) yet for a real Postgres rollout.

### 5. Confidence-Based Routing
`TriageDecision.confidence` is captured but unused downstream. A real triage system would route low-confidence decisions to a human queue instead of auto-acting on them.

### 6. Revisit the Batch Path if Gemini Ships a Discounted Batch Endpoint
`batch.py` currently uses bounded live concurrency because Gemini has no direct equivalent of Anthropic's 50%-off async Batches API at the time of writing. `config.py`'s `batch_use_batches_api` flag is already reserved for this.

### 7. Docker Hardening
The current `Dockerfile` has no non-root user and no `HEALTHCHECK` — both are a short addition before this would be production-grade as a container.

---

## Origin

Converted from `Sample_Incident_Triage_Pipeline_Anthropic_Final.ipynb`, a single-notebook prototype using the Anthropic Messages API (`claude-haiku-4-5`) with a hand-rolled retry loop and no output schema validation. Rebuilt file-by-file with Pydantic-first design, then migrated to the Gemini API (`google-genai`) for cost. Provider details: [Gemini API docs](https://ai.google.dev/gemini-api/docs) · [google-genai SDK](https://github.com/googleapis/python-genai).
