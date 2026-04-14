"""Tests for authenticated resume profile endpoints."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient

from conftest import VALID_SIGNUP

pytestmark = pytest.mark.asyncio


class TestResumeProfileStatus:
    """GET /api/v1/profile/resume."""

    async def test_requires_authentication(self, client: AsyncClient) -> None:
        """Reject unauthenticated requests."""
        response = await client.get("/api/v1/profile/resume")
        assert response.status_code == 401

    async def test_returns_status_for_authenticated_user(
        self, client: AsyncClient
    ) -> None:
        """Return the current user's resume status."""
        signup = await client.post("/api/v1/auth/signup", json=VALID_SIGNUP)
        token = signup.json()["access_token"]

        with patch(
            "web_backend.api.v1.profile.get_resume_profile_status",
            new=AsyncMock(
                return_value=SimpleNamespace(
                    has_resume=True,
                    member_id=1234,
                    last_uploaded_at=None,
                )
            ),
        ):
            response = await client.get(
                "/api/v1/profile/resume",
                headers={"Authorization": f"Bearer {token}"},
            )

        assert response.status_code == 200
        assert response.json()["has_resume"] is True
        assert response.json()["member_id"] == 1234


class TestResumeProfileUpload:
    """POST /api/v1/profile/resume."""

    async def test_upload_requires_authentication(self, client: AsyncClient) -> None:
        """Reject unauthenticated uploads."""
        response = await client.post(
            "/api/v1/profile/resume",
            files={"file": ("resume.pdf", b"pdf-bytes", "application/pdf")},
        )
        assert response.status_code == 401

    async def test_upload_returns_updated_status(self, client: AsyncClient) -> None:
        """Return the updated resume profile state after upload."""
        signup = await client.post("/api/v1/auth/signup", json=VALID_SIGNUP)
        token = signup.json()["access_token"]

        with patch(
            "web_backend.api.v1.profile.upload_resume_for_user",
            new=AsyncMock(
                return_value=SimpleNamespace(
                    action="updated",
                    has_resume=True,
                    member_id=9876,
                    last_uploaded_at=None,
                )
            ),
        ):
            response = await client.post(
                "/api/v1/profile/resume",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("resume.pdf", b"pdf-bytes", "application/pdf")},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["action"] == "updated"
        assert body["has_resume"] is True
        assert body["member_id"] == 9876

    async def test_upload_maps_validation_errors(self, client: AsyncClient) -> None:
        """Translate service validation errors into HTTP 400."""
        signup = await client.post("/api/v1/auth/signup", json=VALID_SIGNUP)
        token = signup.json()["access_token"]

        with patch(
            "web_backend.api.v1.profile.upload_resume_for_user",
            new=AsyncMock(side_effect=ValueError("Only PDF and DOCX are supported.")),
        ):
            response = await client.post(
                "/api/v1/profile/resume",
                headers={"Authorization": f"Bearer {token}"},
                files={"file": ("resume.txt", b"text", "text/plain")},
            )

        assert response.status_code == 400
        assert "PDF" in response.json()["detail"]
