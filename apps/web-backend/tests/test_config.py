"""Tests for backend settings parsing."""

import json

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
