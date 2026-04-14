"""Schemas for authenticated profile resume endpoints."""

from datetime import datetime

from pydantic import BaseModel


class ResumeProfileStatusResponse(BaseModel):
    """Resume/profile status for the current authenticated user."""

    has_resume: bool
    member_id: int | None
    last_uploaded_at: datetime | None


class ResumeProfileUploadResponse(ResumeProfileStatusResponse):
    """Resume upload result for the current authenticated user."""

    action: str
