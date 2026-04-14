"""Schemas for semantic engineer-ticket recommendation endpoints."""

import uuid
from datetime import date

from pydantic import BaseModel


class RecommendedEngineerResponse(BaseModel):
    """One engineer recommendation for a ticket."""

    member_id: int
    user_id: uuid.UUID
    username: str
    first_name: str
    last_name: str
    email: str
    active_ticket_count: int
    capacity_score: float
    has_capacity: bool
    semantic_similarity: float
    lexical_score: float
    recommendation_score: float


class TicketEngineerRecommendationsResponse(BaseModel):
    """Engineer recommendations for a project ticket."""

    ticket_key: str
    recommendations: list[RecommendedEngineerResponse]


class RecommendedTicketResponse(BaseModel):
    """One ticket recommendation for an engineer."""

    ticket_key: str
    title: str
    description: str | None
    priority: str
    type: str
    labels: list[str]
    due_date: date | None
    semantic_similarity: float
    lexical_score: float
    recommendation_score: float
    assignee_id: uuid.UUID | None
    assignee_name: str | None
    column_name: str


class EngineerTicketRecommendationsResponse(BaseModel):
    """Ticket recommendations for a project engineer."""

    user_id: uuid.UUID
    username: str
    first_name: str
    last_name: str
    active_ticket_count: int
    has_capacity: bool
    recommendations: list[RecommendedTicketResponse]
