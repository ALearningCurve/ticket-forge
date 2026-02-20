from pydantic import BaseModel, Field
from typing import List, Optional


class ColdStartRequest(BaseModel):
    """POST body for single resume cold start."""

    resume_file_path: str = Field(..., description="Local path to the resume file (PDF/DOCX)")
    github_username: str = Field(..., description="GitHub username of the engineer")
    full_name: str = Field(..., description="Full name of the engineer")


class ColdStartResponse(BaseModel):
    """Response after profile creation/update."""

    member_id: str
    action: str  # "created" or "updated"
    github_username: str
    full_name: str
    keywords: List[str]
    confidence: float
    experience_weight: float


class ColdStartBatchRequest(BaseModel):
    """POST body for batch resume processing."""

    resume_dir: str = Field(..., description="Directory containing resume files")
    username_map: Optional[dict] = Field(
        None,
        description="Mapping from filename (without extension) to github username",
    )


class ColdStartBatchResponse(BaseModel):
    total_processed: int
    results: List[ColdStartResponse]