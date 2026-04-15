"""Tests for backend settings parsing."""

import json
from pathlib import Path

from web_backend.config import Settings


def test_settings_accepts_json_encoded_cors_origins_from_env(monkeypatch) -> None:
    """Cloud Run env vars should support JSON-encoded CORS origins."""
    monkeypatch.setenv(
        "CORS_ORIGINS",
        json.dumps(
            [
                "https://ticketforge-frontend-r5ebf6yyyq-ue.a.run.app",
                "https://ticketforge.example.com",
            ]
        ),
    )

    settings = Settings()

    assert settings.cors_origins == [
        "https://ticketforge-frontend-r5ebf6yyyq-ue.a.run.app",
        "https://ticketforge.example.com",
    ]


def test_settings_loads_repo_root_env_file_from_any_cwd(monkeypatch) -> None:
    """Backend settings should read the repo-root .env regardless of cwd."""
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.chdir(Path("/home/will/ticket-forge/apps/web-backend"))

    settings = Settings()

    assert settings.mlflow_tracking_uri == (
        "https://mlflow-tracking-994028410655.us-east1.run.app"
    )
