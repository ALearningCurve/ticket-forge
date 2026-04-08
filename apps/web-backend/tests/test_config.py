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


def test_settings_prefers_database_url_over_component_fields(monkeypatch) -> None:
    """A secret-provided DATABASE_URL should override reconstructed DB settings."""
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://ticketforge:prodpass123@10.3.0.3:5432/ticketforge",
    )
    monkeypatch.setenv("DATABASE_HOST", "localhost")
    monkeypatch.setenv("DATABASE_PORT", "5433")
    monkeypatch.setenv("DATABASE_NAME", "local")
    monkeypatch.setenv("DATABASE_USER", "localuser")
    monkeypatch.setenv("DATABASE_PASSWORD", "root")

    settings = Settings()

    assert settings.database_url == (
        "postgresql+asyncpg://ticketforge:prodpass123@10.3.0.3:5432/ticketforge"
    )
    assert settings.resolved_database_url == settings.database_url
