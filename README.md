# Incident Triage

Production build of the notebook incident-triage pipeline, on the **Gemini API**
(`google-genai`). Messy alert text in -> `System` / `Severity` / `Action`
decision out, via structured output validated by Pydantic.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp src/.env.example src/.env # then edit src/.env with a real GEMINI_API_KEY
```

Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey).
The SDK reads `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) from the environment —
it is never read by `Settings`, and never appears in code.

## Run the tests

```bash
python -m pytest src/triage/ -v
```

147 tests, no network access or API key required — every Gemini call is mocked.

## Run the service

```bash
cd src/triage
uvicorn api:app --reload
```

- `POST /triage` — `{"incident_id": "...", "complaint": "..."}` -> `TriageResult`
- `GET /triage/{incident_id}` — fetch a stored result
- `POST /triage/batch` — a list of the same request body -> `202` with a `BatchJob` (runs in the background)
- `GET /triage/batch/{job_id}` — poll job status
- `GET /healthz`

## Run the original 55-incident sample batch

```bash
cd src/triage
python run_sample_batch.py
```

Reproduces the notebook's batch run against the new pipeline and writes
`triage_results.json` (now with real structured `decision` objects, not
pipe-delimited text).

## Layout

| File | Responsibility |
|---|---|
| `models.py` | Pydantic domain models (locked, provider-agnostic) |
| `config.py` | `pydantic-settings`; credential is never modelled here |
| `errors.py` | Gemini exception -> `ErrorInfo` classification |
| `client.py` | One Gemini call: structured output, usage, blocked-response detection |
| `service.py` | Retry orchestration, sync + async |
| `batch.py` | Bounded-concurrency fan-out over `service.triage_async` |
| `store.py` | SQLModel persistence (SQLite by default) |
| `api.py` | FastAPI app |

## Verified live (2026-09-17)

Ran `run_sample_batch.py` against the real Gemini API: **55/55 succeeded**,
sensible classifications, ~116 tokens/incident. Two real integration issues
surfaced and were fixed in `client.py`:

- **Gemini's `response_schema` is a constrained OpenAPI-3.0 subset, not full
  JSON Schema** — it rejects `additionalProperties` outright (400), which
  `TriageDecision.model_json_schema()` includes because of `extra="forbid"`.
  Fixed by building the Gemini schema by hand from the `System`/`Severity`/
  `Action` enums (`client._TRIAGE_SCHEMA`) instead of passing the Pydantic
  class to `response_schema`. `.parsed` now comes back as a plain `dict`,
  which `_interpret()` validates into `TriageDecision` itself.
- **2.5 Flash's "thinking" is on by default and is billed out of
  `max_output_tokens`** — at a 256-token budget it consumed almost the whole
  thing, truncating the actual JSON answer. Fixed with
  `thinking_config=types.ThinkingConfig(thinking_budget=0)` — there's no real
  reasoning needed for this classification task.
- Also: `src/.env` was never actually loaded — `Settings` deliberately never
  models the API key (Option A), but nothing called `load_dotenv()` either,
  and `env_file="."env"` was resolved relative to cwd rather than a fixed
  path. Fixed in `config.py` by loading `.env` from a path resolved off
  `__file__`, so it works regardless of where you run from.

## Known follow-ups

- `batch_use_batches_api` in `config.py` is reserved but unused — the batch
  path is bounded live concurrency, not a discounted async batch endpoint.
- Docker image has no non-root user / healthcheck yet.
