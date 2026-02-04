# Web Backend

FastAPI web service for serving ML model predictions and handling business logic.

## Overview

RESTful API backend that:
- Serves predictions from trained ML models
- Handles ticket assignment recommendations
- Provides health checks and status endpoints

## Quick Start

### Run Development Server

```bash
# From repo root
cd apps/web-backend
uv run uvicorn web_backend.main:app --reload
```

Server will be available at `http://localhost:8000`

### API Documentation

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Project Structure

```
web-backend/
├── web_backend/
│   ├── main.py           # FastAPI application and endpoints
│   └── __init__.py
├── tests/
│   └── test_main.py      # API tests
├── pyproject.toml        # Dependencies and configuration
└── README.md
```

## Dependencies

- **FastAPI** - Modern web framework
- **Pydantic** - Data validation
- **ml-core** - ML utilities and models

See `pyproject.toml` for exact versions.

## Testing

```bash
just pytest apps/web-backend/tests/
```
