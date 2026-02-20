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


@router.post("/", response_model=ColdStartResponse)
def create_coldstart_profile(req: ColdStartRequest):
    """Process a single resume and upsert the engineer profile into Postgres."""

    file_path = Path(req.resume_file_path)
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

    dir_path = Path(req.resume_dir)
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