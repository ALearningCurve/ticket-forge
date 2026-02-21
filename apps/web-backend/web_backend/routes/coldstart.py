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
    """Get the base directory under which resume paths must reside."""
    base_dir = os.environ.get("RESUME_BASE_DIR")
    if not base_dir:
        raise HTTPException(status_code=500, detail="RESUME_BASE_DIR not configured")
    try:
        resolved_base = Path(base_dir).resolve(strict=True)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail=f"RESUME_BASE_DIR does not exist: {base_dir}",
        )
    if not resolved_base.is_dir():
        raise HTTPException(
            status_code=500,
            detail=f"RESUME_BASE_DIR is not a directory: {base_dir}",
        )
    return resolved_base


def _resolve_under_base(base_dir: Path, user_path: str, kind: str) -> Path:
    """
    Resolve a user-supplied path under a trusted base directory.

    Raises HTTPException(400) if the resolved path escapes the base directory.
    """
    candidate = (base_dir / user_path).resolve()
    try:
        # Python 3.9+: Path.is_relative_to
        is_within_base = candidate.is_relative_to(base_dir)  # type: ignore[attr-defined]
    except AttributeError:
        # Fallback for older Python versions
        try:
            candidate.relative_to(base_dir)
            is_within_base = True
        except ValueError:
            is_within_base = False
    if not is_within_base:
        raise HTTPException(
            status_code=400,
            detail=f"{kind} is outside of the allowed directory",
        )
    return candidate


def _get_resume_base_dir() -> Path:
    """Return the base directory under which resume paths must reside."""
    base_dir = os.environ.get("RESUME_BASE_DIR")
    if not base_dir:
    base_dir = _get_resume_base_dir()
    file_path = _resolve_under_base(base_dir, req.resume_file_path, "Resume file path")
    base_path = Path(base_dir).resolve()
    if not base_path.exists() or not base_path.is_dir():
        raise HTTPException(
            status_code=500,
            detail=f"Resume base directory invalid: {base_path}",
        )
    return base_path


@router.post("/", response_model=ColdStartResponse)
def create_coldstart_profile(req: ColdStartRequest):
    """Process a single resume and upsert the engineer profile into Postgres."""

    base_dir = _get_resume_base_dir()
    try:
        # Treat the provided path as relative to the configured base directory.
        requested_path = Path(req.resume_file_path)
        if requested_path.is_absolute():
            raise HTTPException(
                status_code=400,
                detail="Absolute resume paths are not allowed",
            )
        file_path = (base_dir / requested_path).resolve()
    except HTTPException:
        # Re-raise HTTPExceptions unchanged.
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid resume file path",
        )

    # Ensure the resolved path is still within the base directory.
    if not str(file_path).startswith(str(base_dir) + os.sep) and file_path != base_dir:
        raise HTTPException(
            status_code=400,
            detail="Resume file path is outside the allowed directory",
        )
    base_dir = _get_resume_base_dir()
    dir_path = _resolve_under_base(base_dir, req.resume_dir, "Resume directory")
    if not file_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"Resume file not found: {req.resume_file_path}",
        )
    if not file_path.is_file():
        raise HTTPException(
            status_code=400,
            detail=f"Path is not a file: {req.resume_file_path}",
        )

    mgr = _get_manager()

    try:
        profile = mgr.process_resume_file(
            file_path=str(file_path),
            github_username=req.github_username,
            full_name=req.full_name,
        )
        result = mgr.save_profile(profile)
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

    base_dir = _get_resume_base_dir()
    try:
        requested_dir = Path(req.resume_dir)
        if requested_dir.is_absolute():
            raise HTTPException(
                status_code=400,
                detail="Absolute resume directories are not allowed",
            )
        dir_path = (base_dir / requested_dir).resolve()
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid resume directory path",
        )

    # Ensure the resolved directory path is still within the base directory.
    if not str(dir_path).startswith(str(base_dir) + os.sep) and dir_path != base_dir:
        raise HTTPException(
            status_code=400,
            detail="Resume directory is outside the allowed base directory",
        )

    if not dir_path.exists() or not dir_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"Resume directory not found: {req.resume_dir}",
        )

    mgr = _get_manager()

    try:
        profiles = mgr.process_directory(
            resume_dir=str(dir_path),
            username_map=req.username_map,
        )
        db_results = mgr.save_profiles(profiles)
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