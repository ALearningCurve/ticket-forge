"""Authenticated profile endpoints."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from web_backend.database import get_db
from web_backend.models.user import AuthUser
from web_backend.schemas.profile import (
    ResumeProfileStatusResponse,
    ResumeProfileUploadResponse,
)
from web_backend.security.dependencies import get_current_user
from web_backend.services.profile_resume import (
    get_resume_profile_status,
    upload_resume_for_user,
)

router = APIRouter(prefix="/profile", tags=["Profile"])


@router.get("/resume", response_model=ResumeProfileStatusResponse)
async def get_resume_status(
    current_user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ResumeProfileStatusResponse:
    """Return the current user's resume/profile status."""
    status_result = await get_resume_profile_status(db, current_user)
    return ResumeProfileStatusResponse(
        has_resume=status_result.has_resume,
        member_id=status_result.member_id,
        last_uploaded_at=status_result.last_uploaded_at,
    )


@router.post("/resume", response_model=ResumeProfileUploadResponse)
async def upload_resume(
    file: UploadFile = File(...),
    current_user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> ResumeProfileUploadResponse:
    """Upload a resume and update the current user's ML profile."""
    try:
        file_bytes = await file.read()
        result = await upload_resume_for_user(
            db,
            current_user,
            filename=file.filename or "",
            file_bytes=file_bytes,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    return ResumeProfileUploadResponse(
        action=result.action,
        has_resume=result.has_resume,
        member_id=result.member_id,
        last_uploaded_at=result.last_uploaded_at,
    )
