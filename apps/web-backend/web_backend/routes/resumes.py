"""Cold-start routes.

Provides a single batch upload endpoint that accepts resume content
(base64-encoded) and forwards it to an Airflow DAG for asynchronous
processing.  A status-polling endpoint is also provided.

Ticket-assignee profile creation is handled in the ETL pipeline
(``training.etl.ingest.resume.coldstart``) – *not* via the web API.
"""

from fastapi import APIRouter, HTTPException

from web_backend.models.resume import (
    ResumeUploadBatchRequest,
)
from web_backend.models.airflow import PipelineStatusResponse, PipelineTriggerResponse
from web_backend.services.airflow import (
    get_dag_run_status,
    trigger_resume_ingest_batch,
)

router = APIRouter(prefix="/resumes", tags=["Cold Start"])


# ------------------------------------------------------------------ #
#  Batch resume upload  →  Airflow pipeline
# ------------------------------------------------------------------ #


@router.post("/upload", response_model=PipelineTriggerResponse)
def upload_resumes(req: ResumeUploadBatchRequest) -> PipelineTriggerResponse:
    """Accept a batch of resumes and trigger an Airflow DAG run.

    Each resume is sent as base64-encoded content together with a
    ``github_username`` and optional ``full_name``.  The payload is
    forwarded to the Airflow DAG's ``conf`` so the ETL pipeline can
    decode and process each file.
    """
    if not req.resumes:
        raise HTTPException(status_code=400, detail="No resumes provided")

    items = [
        {
            "filename": r.filename,
            "content_base64": r.content_base64,
            "github_username": r.github_username,
            "full_name": r.full_name,
        }
        for r in req.resumes
    ]

    try:
        result = trigger_resume_ingest_batch(items)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    return PipelineTriggerResponse(
        dag_id=result.dag_id,
        run_id=result.run_id,
        status=result.status.value,
        triggered_at=result.triggered_at,
    )


# ------------------------------------------------------------------ #
#  Pipeline status polling
# ------------------------------------------------------------------ #


@router.get("/status/{run_id}", response_model=PipelineStatusResponse)
def get_pipeline_status(run_id: str) -> PipelineStatusResponse:
    """Poll the status of a previously triggered pipeline run."""
    result = get_dag_run_status(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Run ID not found")
    return PipelineStatusResponse(
        dag_id=result.dag_id,
        run_id=result.run_id,
        status=result.status.value,
        conf=result.conf,
        triggered_at=result.triggered_at,
    )
