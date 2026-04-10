"""Recommendation endpoints for engineer-ticket matching."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from web_backend.database import get_db
from web_backend.models.user import AuthUser
from web_backend.schemas.recommendations import (
    EngineerTicketRecommendationsResponse,
    TicketEngineerRecommendationsResponse,
)
from web_backend.security.dependencies import get_current_user
from web_backend.services.recommendations import (
    recommend_engineers_for_ticket,
    recommend_tickets_for_engineer,
)

router = APIRouter(prefix="/projects/{slug}", tags=["Recommendations"])


@router.get(
    "/tickets/{ticket_key}/recommendations/engineers",
    response_model=TicketEngineerRecommendationsResponse,
)
async def get_ticket_engineer_recommendations(
    slug: str,
    ticket_key: str,
    current_user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> TicketEngineerRecommendationsResponse:
    """Return engineer recommendations for a project ticket."""

    try:
        return await recommend_engineers_for_ticket(
            db,
            slug,
            ticket_key,
            current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc


@router.get(
    "/members/{user_id}/recommendations/tickets",
    response_model=EngineerTicketRecommendationsResponse,
)
async def get_engineer_ticket_recommendations(
    slug: str,
    user_id: uuid.UUID,
    current_user: AuthUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> EngineerTicketRecommendationsResponse:
    """Return ticket recommendations for a project engineer."""

    try:
        return await recommend_tickets_for_engineer(
            db,
            slug,
            user_id,
            current_user.id,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
