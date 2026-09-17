FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/triage/ ./src/triage/

WORKDIR /app/src/triage

# GEMINI_API_KEY and any TRIAGE_* overrides are injected at runtime
# (docker run -e / --env-file), never baked into the image.
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
