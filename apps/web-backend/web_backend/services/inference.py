"""Production inference service for ticket size prediction."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from time import perf_counter
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
from ml_core.embeddings import get_embedding_service
from ml_core.features import REPO_FEATURE_ORDER, TOP_50_LABELS
from ml_core.keywords import get_keyword_extractor
from ml_core.retrieval import vector_to_pgvector_text
from mlflow.tracking import MlflowClient
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from web_backend.config import get_settings
from web_backend.models.inference import InferenceEvent
from web_backend.schemas.inference import (
    InferenceMonitoringRecordResponse,
    InferenceFeatureSummaryResponse,
    InferenceModelMetadataResponse,
    TicketSizePredictionRequest,
    TicketSizePredictionResponse,
)

logger = logging.getLogger(__name__)

_SIZE_BUCKET_LABELS: dict[int, str] = {
    0: "S",
    1: "M",
    2: "L",
    3: "XL",
}
_DEFAULT_SIZE_POINTS: dict[str, float] = {
    "S": 1.0,
    "M": 2.0,
    "L": 3.0,
    "XL": 4.0,
}
_CODE_BLOCK_RE = re.compile(r"```(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_MAX_TTA_HOURS = 720.0
_SEMANTIC_RANK_DENOMINATOR_OFFSET = 2


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """Loaded production model and its serving metadata."""

    estimator: Any
    selector: str
    tracking_uri: str | None
    model_name: str
    model_stage: str | None
    model_version: str | None
    model_run_id: str | None


@dataclass(frozen=True, slots=True)
class SemanticSizeEstimate:
    """Semantic size estimate derived from similar tickets."""

    average_points: float
    sample_count: int
    bucket: str


def _tracking_uri() -> str | None:
    """Return the explicit tracking URI configured for backend serving."""
    return get_settings().mlflow_tracking_uri


def _model_cache_key() -> tuple[str | None, str | None, str]:
    """Build a stable cache key for the loaded model bundle."""
    settings = get_settings()
    return (
        settings.mlflow_tracking_uri,
        settings.serving_model_version,
        settings.mlflow_model_stage,
    )


def _truncate_code_block(code: str) -> str:
    """Truncate long fenced code blocks before embedding."""
    lines = code.strip().splitlines()
    if len(lines) <= 15:
        return "\n".join(lines)
    if len(lines) <= 50:
        return "\n".join(lines[:5] + ["..."] + lines[-5:])
    return "\n".join(lines[:10] + ["..."] + lines[-10:])


def _find_balanced_section_end(
    text: str,
    start_index: int,
    opening: str,
    closing: str,
) -> int:
    """Return the closing delimiter index for a balanced markdown section."""
    depth = 1
    index = start_index
    while index < len(text):
        char = text[index]
        if char == "\\":
            index += 2
            continue
        if char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _strip_markdown_links(text: str) -> str:
    """Strip markdown image/link syntax without regex backtracking."""
    chunks: list[str] = []
    index = 0

    while index < len(text):
        is_image = text.startswith("![", index)
        if is_image or text[index] == "[":
            label_start = index + (2 if is_image else 1)
            label_end = _find_balanced_section_end(text, label_start, "[", "]")
            if (
                label_end != -1
                and label_end + 1 < len(text)
                and text[label_end + 1] == "("
            ):
                target_end = _find_balanced_section_end(
                    text,
                    label_end + 2,
                    "(",
                    ")",
                )
                if target_end != -1:
                    if not is_image:
                        chunks.append(text[label_start:label_end])
                    index = target_end + 1
                    continue

        chunks.append(text[index])
        index += 1

    return "".join(chunks)


def _normalize_ticket_text(title: str, body: str) -> str:
    """Normalize markdown-heavy ticket text into a model-friendly payload."""
    safe_body = body or ""

    def _replace_code_block(match: re.Match[str]) -> str:
        return _truncate_code_block(match.group(1))

    safe_body = _CODE_BLOCK_RE.sub(_replace_code_block, safe_body)
    safe_body = _strip_markdown_links(safe_body)
    safe_body = _INLINE_CODE_RE.sub(r"\1", safe_body)
    safe_body = re.sub(r"[>#*_~-]", " ", safe_body)
    safe_body = re.sub(r"\n{3,}", "\n\n", safe_body)
    safe_body = re.sub(r"\s+", " ", safe_body).strip()
    combined = f"{title.strip()}\n\n{safe_body}".strip()
    return combined[:4000]


def _coerce_datetime(value: datetime | None) -> datetime | None:
    """Normalize possibly-naive datetimes to UTC-aware values."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _time_to_assignment_hours(
    created_at: datetime | None,
    assigned_at: datetime | None,
) -> float:
    """Compute time to assignment in hours with the training-time cap."""
    created = _coerce_datetime(created_at)
    assigned = _coerce_datetime(assigned_at)
    if created is None or assigned is None:
        return 0.0
    diff_hours = max((assigned - created).total_seconds() / 3600.0, 0.0)
    return float(min(diff_hours, _MAX_TTA_HOURS))


def _extract_keywords(text: str) -> list[str]:
    """Return the top technical keywords detected in the normalized text."""
    return get_keyword_extractor().extract(text, top_n=10)


def _embed_text(text: str) -> np.ndarray:
    """Generate the 384-dim embedding used during training."""
    return get_embedding_service(model_name="all-MiniLM-L6-v2").embed_text(text)


def _points_for_bucket(bucket: str, size_points_map: dict[str, float]) -> float:
    """Return the point value for a size bucket."""
    return float(size_points_map.get(bucket, _DEFAULT_SIZE_POINTS.get(bucket, 0.0)))


def _bucket_for_points(points: float, size_points_map: dict[str, float]) -> str:
    """Map a numeric estimate back to the nearest size bucket."""
    return min(
        size_points_map.items(),
        key=lambda item: (abs(item[1] - points), item[1]),
    )[0]


def _ordinal_size_points_map() -> dict[str, float]:
    """Return the fixed ordinal scale for size blending."""
    return {"S": 1.0, "M": 2.0, "L": 3.0, "XL": 4.0}


def _semantic_rank_weight(rank: int) -> float:
    """Return a softened, monotonically decreasing semantic neighbor weight."""
    denominator = rank + _SEMANTIC_RANK_DENOMINATOR_OFFSET
    if denominator <= 0:
        return 0.0
    return 1.0 / float(denominator)


def _blend_size_points(
    model_points: float,
    semantic_points: float | None,
    model_weight: float,
) -> float:
    """Blend model and semantic points into a single size estimate."""
    if semantic_points is None:
        return model_points
    clamped_weight = max(0.0, min(model_weight, 1))
    semantic_weight = 1.0 - clamped_weight
    return round(
        (model_points * clamped_weight) + (semantic_points * semantic_weight),
        6,
    )


async def _estimate_semantic_size(
    db: AsyncSession,
    payload: TicketSizePredictionRequest,
    *,
    ticket_vector: np.ndarray,
) -> SemanticSizeEstimate | None:
    """Estimate ticket size from the five most similar sized tickets."""
    if payload.project_id is None:
        return None

    size_points_map = _ordinal_size_points_map()

    rows: list[dict[str, Any]] = [
        dict(row)
        for row in (
            await db.execute(
                text(
                    """
                                        WITH candidate_neighbors AS MATERIALIZED (
                      SELECT
                        pt.size_bucket,
                                                t.ticket_vector <=> CAST(:ticket_vector AS vector) AS distance
                      FROM project_tickets pt
                      JOIN tickets t
                        ON t.ticket_id = pt.ticket_key
                      WHERE pt.project_id = :project_id
                        AND pt.size_bucket IS NOT NULL
                                                AND (
                                                    CAST(:ticket_id AS uuid) IS NULL
                                                    OR pt.id <> CAST(:ticket_id AS uuid)
                                                )
                    )
                    SELECT
                                            size_bucket
                                        FROM candidate_neighbors
                                        ORDER BY distance ASC
                                        LIMIT 5
                    """
                ),
                {
                    "project_id": payload.project_id,
                    "ticket_id": payload.ticket_id,
                    "ticket_vector": vector_to_pgvector_text(ticket_vector.tolist()),
                },
            )
        )
        .mappings()
        .all()
    ]

    if not rows:
        logger.info(
            "Semantic size estimate skipped project_id=%s ticket_id=%s no_neighbors=true",
            payload.project_id,
            payload.ticket_id,
        )
        return None

    weighted_total = 0.0
    weight_sum = 0.0
    debug_neighbors: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        bucket = str(row["size_bucket"])
        if rank <= 0:
            continue
        points = _points_for_bucket(bucket, size_points_map)
        weight = _semantic_rank_weight(rank)
        weighted_total += points * weight
        weight_sum += weight
        debug_neighbors.append(
            {
                "bucket": bucket,
                "rank": rank,
                "points": points,
                "weight": round(weight, 6),
            }
        )

    if weight_sum <= 0.0:
        logger.info(
            "Semantic size estimate skipped project_id=%s ticket_id=%s weight_sum=0",
            payload.project_id,
            payload.ticket_id,
        )
        return None

    average_points_float = round(weighted_total / weight_sum, 6)
    bucket = _bucket_for_points(average_points_float, size_points_map)
    logger.info(
        (
            "Semantic size estimate project_id=%s ticket_id=%s neighbors=%s "
            "weighted_points=%.6f bucket=%s"
        ),
        payload.project_id,
        payload.ticket_id,
        debug_neighbors,
        average_points_float,
        bucket,
    )

    return SemanticSizeEstimate(
        average_points=average_points_float,
        sample_count=len(debug_neighbors),
        bucket=bucket,
    )


def _build_feature_vector(
    payload: TicketSizePredictionRequest,
) -> tuple[np.ndarray, InferenceFeatureSummaryResponse]:
    """Build the exact feature vector expected by the trained classifier."""
    normalized_text = _normalize_ticket_text(payload.title, payload.body)
    keywords = _extract_keywords(normalized_text)
    embedding = _embed_text(normalized_text).astype(np.float32)

    repo = (payload.repo or "").strip()
    labels = [label.strip() for label in payload.labels if label.strip()]
    ticket_labels = set(labels)
    repo_one_hot = [
        1.0 if repo == repo_name else 0.0 for repo_name in REPO_FEATURE_ORDER
    ]
    label_one_hot = [1.0 if label in ticket_labels else 0.0 for label in TOP_50_LABELS]
    title_length = len(payload.title or "")
    body_length = len(payload.body or "")
    keyword_count = len(keywords)
    time_to_assignment = _time_to_assignment_hours(
        payload.created_at, payload.assigned_at
    )

    engineered_features = np.array(
        repo_one_hot
        + label_one_hot
        + [
            float(payload.comments_count),
            float(payload.historical_avg_completion_hours),
            float(keyword_count),
            float(title_length),
            float(body_length),
            float(time_to_assignment),
        ],
        dtype=np.float32,
    )
    feature_vector = np.nan_to_num(
        np.hstack([embedding, engineered_features]),
        nan=0.0,
    ).reshape(1, -1)
    summary = InferenceFeatureSummaryResponse(
        repo=repo,
        labels=labels,
        keyword_count=keyword_count,
        comments_count=payload.comments_count,
        historical_avg_completion_hours=float(payload.historical_avg_completion_hours),
        title_length=title_length,
        body_length=body_length,
        time_to_assignment_hours=round(time_to_assignment, 6),
    )
    return feature_vector, summary


def _load_version_metadata(client: MlflowClient) -> tuple[str, str | None, str | None]:
    """Resolve the exact model selector and metadata for serving."""
    settings = get_settings()
    if settings.serving_model_version:
        version = client.get_model_version(
            settings.mlflow_registered_model_name,
            settings.serving_model_version,
        )
        return settings.serving_model_version, None, version.run_id

    versions = client.get_latest_versions(
        settings.mlflow_registered_model_name,
        stages=[settings.mlflow_model_stage],
    )
    if not versions:
        msg = (
            f"No model version found for "
            f"{settings.mlflow_registered_model_name}:{settings.mlflow_model_stage}"
        )
        raise RuntimeError(msg)

    version = versions[0]
    return str(version.version), settings.mlflow_model_stage, version.run_id


@lru_cache(maxsize=1)
def _load_model(_cache_key: tuple[str | None, str | None, str]) -> LoadedModel:
    """Load the currently configured model from MLflow once per process."""
    settings = get_settings()
    tracking_uri = _tracking_uri()
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri) if tracking_uri else MlflowClient()
    resolved_version, resolved_stage, run_id = _load_version_metadata(client)
    selector = settings.serving_model_version or settings.mlflow_model_stage
    model_uri = f"models:/{settings.mlflow_registered_model_name}/{selector}"
    estimator = mlflow.sklearn.load_model(model_uri)

    logger.info(
        "Loaded serving model selector=%s version=%s tracking_uri=%s",
        selector,
        resolved_version,
        tracking_uri,
    )
    return LoadedModel(
        estimator=estimator,
        selector=selector,
        tracking_uri=tracking_uri,
        model_name=settings.mlflow_registered_model_name,
        model_stage=resolved_stage,
        model_version=resolved_version,
        model_run_id=run_id,
    )


def get_loaded_model() -> LoadedModel:
    """Return the cached serving model bundle."""
    return _load_model(_model_cache_key())


def _softmax(values: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def _class_probabilities(
    estimator: Any, features: np.ndarray
) -> tuple[dict[str, float], float]:
    """Return normalized class probabilities and top confidence."""
    class_ids = list(getattr(estimator, "classes_", range(len(_SIZE_BUCKET_LABELS))))
    probability_map = {label: 0.0 for label in _SIZE_BUCKET_LABELS.values()}

    if hasattr(estimator, "predict_proba"):
        probabilities = np.asarray(estimator.predict_proba(features))[0]
    elif hasattr(estimator, "decision_function"):
        decision_scores = np.asarray(estimator.decision_function(features))
        if decision_scores.ndim == 1:
            decision_scores = np.stack([-decision_scores, decision_scores], axis=1)
        probabilities = _softmax(decision_scores[0])
    else:
        predicted_class = int(np.asarray(estimator.predict(features))[0])
        probabilities = np.zeros(len(class_ids), dtype=np.float64)
        if predicted_class in class_ids:
            probabilities[class_ids.index(predicted_class)] = 1.0

    for index, class_id in enumerate(class_ids):
        bucket = _SIZE_BUCKET_LABELS.get(int(class_id), str(class_id))
        if index < len(probabilities):
            probability_map[bucket] = round(float(probabilities[index]), 6)

    confidence = max(probability_map.values(), default=0.0)
    return probability_map, round(confidence, 6)


def _request_fingerprint(
    payload: TicketSizePredictionRequest,
    summary: InferenceFeatureSummaryResponse,
) -> str:
    """Create a deterministic fingerprint for the inference request."""
    fingerprint_source = {
        "title": payload.title,
        "body": payload.body,
        "repo": payload.repo,
        "issue_type": payload.issue_type,
        "labels": summary.labels,
        "comments_count": payload.comments_count,
        "historical_avg_completion_hours": payload.historical_avg_completion_hours,
        "created_at": payload.created_at.isoformat() if payload.created_at else None,
        "assigned_at": payload.assigned_at.isoformat() if payload.assigned_at else None,
        "project_id": str(payload.project_id) if payload.project_id else None,
        "ticket_id": str(payload.ticket_id) if payload.ticket_id else None,
    }
    encoded = json.dumps(fingerprint_source, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


async def _store_inference_event(
    db: AsyncSession,
    *,
    payload: TicketSizePredictionRequest,
    summary: InferenceFeatureSummaryResponse,
    model: LoadedModel,
    predicted_bucket: str,
    predicted_class: int,
    confidence: float,
    latency_ms: float,
) -> None:
    """Persist inference telemetry without failing the main request path."""
    event = InferenceEvent(
        rail=payload.rail,
        request_fingerprint=_request_fingerprint(payload, summary),
        model_name=model.model_name,
        model_stage=model.model_stage,
        model_version=model.model_version,
        model_run_id=model.model_run_id,
        tracking_uri=model.tracking_uri,
        predicted_bucket=predicted_bucket,
        predicted_class=predicted_class,
        confidence=confidence,
        latency_ms=latency_ms,
        repo=summary.repo or None,
        issue_type=payload.issue_type,
        labels=summary.labels,
        comments_count=summary.comments_count,
        historical_avg_completion_hours=summary.historical_avg_completion_hours,
        keyword_count=summary.keyword_count,
        title_length=summary.title_length,
        body_length=summary.body_length,
        time_to_assignment_hours=summary.time_to_assignment_hours,
    )
    db.add(event)
    try:
        await db.commit()
    except Exception:
        await db.rollback()
        logger.exception("Failed to persist inference event")


def current_model_metadata() -> InferenceModelMetadataResponse:
    """Return metadata for the currently served MLflow model."""
    model = get_loaded_model()
    return InferenceModelMetadataResponse(
        model_name=model.model_name,
        model_stage=model.model_stage,
        model_version=model.model_version,
        model_run_id=model.model_run_id,
        tracking_uri=model.tracking_uri,
        selector=model.selector,
    )


async def predict_ticket_size(
    db: AsyncSession,
    payload: TicketSizePredictionRequest,
) -> TicketSizePredictionResponse:
    """Run live ticket size prediction against the promoted production model."""
    started_at = perf_counter()
    model = get_loaded_model()
    features, summary = _build_feature_vector(payload)
    size_points_map = _ordinal_size_points_map()
    predicted_class = int(np.asarray(model.estimator.predict(features))[0])
    model_bucket = _SIZE_BUCKET_LABELS.get(predicted_class, str(predicted_class))
    class_probabilities, confidence = _class_probabilities(model.estimator, features)

    semantic_estimate = await _estimate_semantic_size(
        db,
        payload,
        ticket_vector=_embed_text(_normalize_ticket_text(payload.title, payload.body)),
    )
    model_points = _points_for_bucket(model_bucket, size_points_map)
    blended_points = _blend_size_points(
        model_points,
        semantic_estimate.average_points if semantic_estimate else None,
        confidence,
    )
    predicted_bucket = _bucket_for_points(blended_points, size_points_map)
    logger.info(
        (
            "Ticket size blend project_id=%s ticket_id=%s model_bucket=%s "
            "model_points=%.6f confidence=%.6f semantic_bucket=%s "
            "semantic_points=%s blended_points=%.6f final_bucket=%s"
        ),
        payload.project_id,
        payload.ticket_id,
        model_bucket,
        model_points,
        confidence,
        semantic_estimate.bucket if semantic_estimate else None,
        (
            f"{semantic_estimate.average_points:.6f}"
            if semantic_estimate is not None
            else None
        ),
        blended_points,
        predicted_bucket,
    )
    latency_ms = round((perf_counter() - started_at) * 1000.0, 3)

    await _store_inference_event(
        db,
        payload=payload,
        summary=summary,
        model=model,
        predicted_bucket=predicted_bucket,
        predicted_class=predicted_class,
        confidence=confidence,
        latency_ms=latency_ms,
    )

    logger.info(
        "Served ticket-size inference model=%s version=%s latency_ms=%.3f bucket=%s",
        model.model_name,
        model.model_version,
        latency_ms,
        predicted_bucket,
    )
    return TicketSizePredictionResponse(
        predicted_bucket=predicted_bucket,
        predicted_class=predicted_class,
        confidence=confidence,
        class_probabilities=class_probabilities,
        latency_ms=latency_ms,
        model=current_model_metadata(),
        features=summary,
        model_bucket=model_bucket,
        semantic_bucket=semantic_estimate.bucket if semantic_estimate else None,
        semantic_average_points=(
            semantic_estimate.average_points if semantic_estimate else None
        ),
        semantic_sample_count=(
            semantic_estimate.sample_count if semantic_estimate else 0
        ),
        blended_points=blended_points,
    )


async def export_recent_inference_records(
    db: AsyncSession,
    *,
    limit: int = 1000,
) -> list[InferenceMonitoringRecordResponse]:
    """Export recent serving telemetry for monitoring workflows."""
    result = await db.execute(
        select(InferenceEvent).order_by(InferenceEvent.created_at.desc()).limit(limit)
    )
    events = result.scalars().all()
    return [
        InferenceMonitoringRecordResponse(
            created_at=event.created_at,
            repo=event.repo,
            issue_type=event.issue_type,
            predicted_bucket=event.predicted_bucket,
            model_version=event.model_version,
            rail=event.rail,
            latency_ms=event.latency_ms,
            confidence=event.confidence,
            comments_count=event.comments_count,
            historical_avg_completion_hours=event.historical_avg_completion_hours,
            keyword_count=event.keyword_count,
            title_length=event.title_length,
            body_length=event.body_length,
            time_to_assignment_hours=event.time_to_assignment_hours,
        )
        for event in events
    ]
