from pydantic import BaseModel, Field
from typing import List, Optional


# ------------------------------------------------------------------ #
#  Resume upload models  (batch-only interface)
# ------------------------------------------------------------------ #


class ResumeUploadItem(BaseModel):
    """A single resume in a batch upload."""

    filename: str = Field(..., description="Original filename (e.g. 'john_doe.pdf')")
    content_base64: str = Field(
        ...,
        description="Base64-encoded file content (PDF or plain-text)",
    )
    github_username: str = Field(..., description="GitHub username for this resume")
    full_name: Optional[str] = Field(
        None, description="Full name of the engineer (optional)"
    )


class ResumeUploadBatchRequest(BaseModel):
    """POST body for batch resume upload → Airflow pipeline."""

    resumes: List[ResumeUploadItem] = Field(
        ..., description="List of resumes to ingest"
    )
