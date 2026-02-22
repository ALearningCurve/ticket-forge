from pydantic import BaseModel


class PipelineTriggerResponse(BaseModel):
    """Response after an Airflow DAG run has been triggered."""

    dag_id: str
    run_id: str
    status: str
    triggered_at: str


class PipelineStatusResponse(BaseModel):
    """Response when polling for a DAG run status."""

    dag_id: str
    run_id: str
    status: str
    conf: dict
    triggered_at: str
