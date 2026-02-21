import os
from pathlib import Path

from fastapi import APIRouter, HTTPException

from training.etl.ingest.resume.coldstart import ColdStartManager
from web_backend.schemas import (
    ColdStartRequest,
    ColdStartResponse,
    ColdStartBatchRequest,
    ColdStartBatchResponse,
)

router = APIRouter(prefix="/coldstart", tags=["Cold Start"])


def _get_manager() -> ColdStartManager:
    """Create a ColdStartManager using DATABASE_URL from env."""
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise HTTPException(status_code=500, detail="DATABASE_URL not configured")
    return ColdStartManager(dsn=dsn)


def _get_resume_base_dir() -> Path:
    """Return the base directory under which resume paths must reside."""
    base_dir = os.environ.get("RESUME_BASE_DIR")
    if not base_dir:
        raise HTTPException(status_code=500, detail="RESUME_BASE_DIR not configured")
    base_path = Path(base_dir).resolve()
    if not base_path.exists() or not base_path.is_dir():
        raise HTTPException(
            status_code=500,
            detail=f"Resume base directory invalid: {base_path}",
        )
    return base_path


def _validate_resume_path(file_path_str: str) -> Path:
    """Validate that a resume file path is inside the allowed base directory."""
    base_dir = _get_resume_base_dir()

    # Only allow filename or relative path — reject absolute paths
    requested = Path(file_path_str)
    if requested.is_absolute():
        raise HTTPException(
            status_code=400,
            detail="Absolute file paths are not allowed. Provide a filename or relative path.",
        )

    # Resolve against base dir and ensure it stays inside
    full_path = (base_dir / requested).resolve()

    if not full_path.is_relative_to(base_dir):
        raise HTTPException(
            status_code=400,
            detail="Resume path is outside the allowed base directory.",
        )
    if not full_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Resume file not found: {requested}",
        )
    if not full_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Path is not a file: {requested}",
        )
    return full_path


def _validate_resume_dir(dir_path_str: str) -> Path:
    """Validate that a resume directory path is inside the allowed base directory."""
    base_dir = _get_resume_base_dir()

    requested = Path(dir_path_str)
    if requested.is_absolute():
        raise HTTPException(
            status_code=400,
            detail="Absolute directory paths are not allowed. Provide a relative path.",
        )

    full_path = (base_dir / requested).resolve()

    if not full_path.is_relative_to(base_dir):
        raise HTTPException(
            status_code=400,
            detail="Resume directory is outside the allowed base directory.",
        )
    if not full_path.exists() or not full_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Resume directory not found: {requested}",
        )
    return full_path


@router.post("/", response_model=ColdStartResponse)
def create_coldstart_profile(req: ColdStartRequest):
    """Process a single resume and upsert the engineer profile into Postgres."""

    file_path = _validate_resume_path(req.resume_file_path)
    mgr = _get_manager()

    try:
        profile = mgr.process_resume_file(
            file_path=str(file_path),
            github_username=req.github_username,
            full_name=req.full_name,
        )
        result = mgr.save_profile(profile)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ColdStartResponse(
        member_id=result["member_id"],
        action=result["action"],
        github_username=req.github_username,
        full_name=req.full_name,
        keywords=profile.keywords,
        confidence=profile.confidence,
        experience_weight=profile.experience_weight,
    )


@router.post("/batch", response_model=ColdStartBatchResponse)
def create_coldstart_batch(req: ColdStartBatchRequest):
    """Process all resumes in a directory and upsert profiles into Postgres."""

    dir_path = _validate_resume_dir(req.resume_dir)
    mgr = _get_manager()

    try:
        profiles = mgr.process_directory(
            resume_dir=str(dir_path),
            username_map=req.username_map,
        )
        db_results = mgr.save_profiles(profiles)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    responses = []
    for profile, db_result in zip(profiles, db_results):
        responses.append(
            ColdStartResponse(
                member_id=db_result["member_id"],
                action=db_result["action"],
                github_username=profile.github_username or "",
                full_name=profile.full_name or "",
                keywords=profile.keywords,
                confidence=profile.confidence,
                experience_weight=profile.experience_weight,
            )
        )

    return ColdStartBatchResponse(
        total_processed=len(responses),
        results=responses,
    )