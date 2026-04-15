# Web Backend

FastAPI web service for serving ML model predictions and handling business logic.

## Overview

RESTful API backend that:
- Serves predictions from trained ML models
- Pins the deployed revision to an exact MLflow Production model version
- Persists inference telemetry for monitoring and drift detection
- Persists AI-estimated ticket size buckets on project board tickets
- Blends model size predictions with the five most semantically similar sized tickets when project context is available, weighted by the model's class confidence
- Handles ticket assignment recommendations
- Provides health checks and status endpoints

## Quick Start

### Run Development Server

```bash
# From repo root
cd apps/web-backend
uv run uvicorn web_backend.main:app --reload
```

The backend loads the repo-root `.env` automatically, so the MLflow tracking
URI and credentials are available even when you start the server from inside
`apps/web-backend/`.

Server will be available at `http://localhost:8000`

### API Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

Key production inference endpoints:

- `GET /health`
- `GET /api/v1/inference/model`
- `POST /api/v1/inference/ticket-size`
- `GET /api/v1/inference/monitoring/export`

Authenticated profile endpoints:

- `GET /api/v1/profile/resume`
- `POST /api/v1/profile/resume`

Authenticated recommendation endpoints:

- `GET /api/v1/projects/{slug}/tickets/{ticket_key}/recommendations/engineers`
- `GET /api/v1/projects/{slug}/members/{user_id}/recommendations/tickets`

Key project ticket sizing behavior:

- `POST /api/v1/projects/{slug}/tickets`
  - auto-predicts a ticket size when no manual bucket is provided
  - blends the model output with a semantic size estimate from the five most relevant historical tickets, weighted by class confidence
- `PATCH /api/v1/projects/{slug}/tickets/{ticket_key}`
  - preserves manual size overrides and re-runs prediction when tickets return to auto mode
- `POST /api/v1/projects/{slug}/tickets/classify-missing`
  - batch-classifies legacy unsized tickets for a project and persists the buckets

## Structure

```
web_backend/
├── routes/            # FastAPI routers (thin handlers)
├── services/          # Business logic and orchestration
├── models/            # Pydantic request/response models
└── main.py            # FastAPI application entry point
```

**Architecture:** Strict separation of concerns following a layered pattern:
- **Routes** — Parse requests, call services, return responses. No business logic.
- **Services** — Orchestrate operations, call `ml-core` utilities, interact with database.
- **Models** — Pydantic schemas for validation. No side effects.

## Dependencies

- **FastAPI** - Modern web framework
- **Pydantic** - Data validation
- **ml-core** - ML utilities and models

See `pyproject.toml` for exact versions.

## Testing

```bash
just pytest apps/web-backend/tests/
```
